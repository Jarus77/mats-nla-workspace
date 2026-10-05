"""``wsbench <command> key=value ...``; every command is a ``pydra.Config`` (fields in
``__init__``, normalised in ``finalize()``); ``--show`` prints the resolved config."""

import json
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pydra

from wsbench import readplan, registry, runner
from wsbench.baselines import lucky_guessing, prompt_only
from wsbench.capable import questions as capable_questions
from wsbench.capable import run as capable_run
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.llm import JudgeConfigError
from wsbench.readouts import convert_gen_dir, convert_read_json
from wsbench.registry import EvalSpec
from wsbench.results import FamilyResult, macro, markdown_table, read_results
from wsbench.runner import FamilyOutcome, parse_opts

EXIT_USAGE = 2
EXIT_JUDGE_CONFIG = 3


def _bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.lower() in ("true", "t", "1", "false", "f", "0"):
        return v.lower() in ("true", "t", "1")
    raise ValueError(f"expected True or False, got {v!r}")


def _int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int | str):
        raise ValueError(f"expected an int, got {v!r}")
    return int(v)


def _ints(v: object) -> list[int] | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        out = [_int(x) for x in v.split(",") if x.strip()]
    elif isinstance(v, Iterable):
        out = [_int(x) for x in v]
    else:
        out = [_int(v)]
    if not out:
        raise ValueError("expected at least one int")
    return out


def _strs(v: object) -> list[str] | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, bool):
        raise ValueError(f"expected a string list, got {v!r}")
    if isinstance(v, Iterable):
        return [str(x) for x in v]
    return [str(v)]


def _path(v: object) -> Path:
    if v is None or isinstance(v, bool) or v == "":
        raise ValueError(f"expected a path, got {v!r}")
    return Path(str(v))


class Command(pydra.Config):
    def execute(self) -> int:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


class JudgeOptions(Command):
    """The judge keys shared by ``judge`` and ``run``; ``runner`` reads them by attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.judge_model = ""
        self.layers = None
        self.items = None
        self.limit = 0
        self.allow_missing = False
        self.concurrency = 64
        self.rpm = 240.0
        self.dry_run = False
        self.opts = None  # comma list of KEY=VALUE family options; a value cannot hold a comma
        self.opt: list[str] = []  # the parsed pairs; filled by finalize

    def finalize(self) -> None:
        self.judge_model = str(self.judge_model or "") or None
        self.layers = _ints(self.layers)
        self.items = _strs(self.items)
        self.limit = _int(self.limit)
        self.allow_missing = _bool(self.allow_missing)
        self.concurrency = _int(self.concurrency)
        self.rpm = float(self.rpm)
        self.dry_run = _bool(self.dry_run)
        self.opt = _strs(self.opts) or []
        self.opts = None


class ListFamilies(Command):
    def execute(self) -> int:
        if not registry.FAMILIES:
            print("no families registered yet")
            return 0
        rows = [
            (
                s.name,
                s.group,
                _n_items(s),
                s.metric,
                s.scorer or s.judge.model,
                s.judge.prompt_version,
                s.calls_per_arm or "?",
                s.sources or "—",
            )
            for s in sorted(registry.FAMILIES.values(), key=lambda s: s.name)
        ]
        head = ("family", "group", "n items", "metric", "judge model", "prompt version")
        head += ("calls/arm (approx)", "credit")
        widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
        for r in [head, *rows]:
            print("  ".join(str(v).ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())
        return 0


class JudgeFamily(JudgeOptions):
    def __init__(self) -> None:
        super().__init__()
        self.family = pydra.REQUIRED
        self.readouts = pydra.REQUIRED
        self.out = ""

    def finalize(self) -> None:
        super().finalize()
        self.family = str(self.family)
        self.readouts = _path(self.readouts)
        self.out = _path(self.out) if self.out else None

    def execute(self) -> int:
        if self.family not in registry.FAMILIES:
            return _unknown(self.family)
        spec = registry.get(self.family)
        out = self.out or Path("outputs") / self.readouts.stem / spec.name
        try:
            opts = parse_opts(self.opt)
            judge = resolve(spec.judge, flag=self.judge_model, env=os.environ)
            runner.judge_family(spec, self, self.readouts, out, judge=judge, opts=opts)
        except JudgeConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return EXIT_JUDGE_CONFIG
        return 0


class RunFamilies(JudgeOptions):
    def __init__(self) -> None:
        super().__init__()
        self.all = False
        self.families = None
        self.readouts_root = pydra.REQUIRED
        self.out = pydra.REQUIRED
        self.family_workers = 3
        self.json = False

    def finalize(self) -> None:
        super().finalize()
        self.all = _bool(self.all)
        self.families = _strs(self.families)
        self.readouts_root = _path(self.readouts_root)
        self.out = _path(self.out)
        self.family_workers = _int(self.family_workers)
        self.json = _bool(self.json)
        if self.all == (self.families is not None):
            raise ValueError("pass exactly one of all=True or families=a,b")

    def execute(self) -> int:
        names = sorted(registry.FAMILIES) if self.all else list(self.families or [])
        for n in names:
            if n not in registry.FAMILIES:
                return _unknown(n)
        specs = [registry.get(n) for n in names]
        started = datetime.now(UTC)
        outcomes, code = runner.run_families(
            specs, self, readouts_root=self.readouts_root, out=self.out
        )
        if not outcomes:  # aborted before any family started (preflight / bad override)
            return code
        results = [o.result for o in outcomes if o.result is not None]
        notes = [o for o in outcomes if o.status != "ok"]
        text = _write_summary(self.out, results, notes)
        manifest = runner.run_manifest(
            outcomes,
            started=started,
            finished=datetime.now(UTC),
            args=self,
            out=self.out,
            readouts_root=self.readouts_root,
        )
        (self.out / "run.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if self.json:
            payload = {
                "families": [r.to_json() for r in results],
                "macro": macro(results),
                "statuses": manifest["families"],
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(text, end="")
            print(f"wrote {self.out / 'summary.md'} and {self.out / 'run.json'}")
        return code


class ReportRuns(Command):
    def __init__(self) -> None:
        super().__init__()
        self.dir = pydra.REQUIRED
        self.json = False
        self.floors = True  # the frozen lucky-guessing floors (evals/baselines) beside each family

    def finalize(self) -> None:
        self.dir = _path(self.dir)
        self.json = _bool(self.json)
        self.floors = _bool(self.floors)

    def execute(self) -> int:
        if not self.dir.is_dir():
            print(f"not a directory: {self.dir}", file=sys.stderr)
            return EXIT_USAGE
        try:
            results = [read_results(p.parent) for p in sorted(self.dir.glob("*/results.json"))]
        except ValueError as e:
            print(f"unreadable results: {e}", file=sys.stderr)
            return EXIT_USAGE
        m = macro(results)
        floors = floor_columns() if self.floors else None
        table = markdown_table(results, m, floors=floors)
        (self.dir / "summary.md").write_text(table, encoding="utf-8")
        if self.json:
            raw = {"lucky_guessing": lucky_guessing.floors(), "prompt_only": prompt_only.floors()}
            out = {"families": [r.to_json() for r in results], "macro": m, "floors": raw}
            print(json.dumps(out))
            return 0
        print(table, end="")
        print(f"macro: value={m['value']} families={m['families']} excluded={m['excluded']}")
        print(f"wrote {self.dir / 'summary.md'}")
        return 0


class ConvertGenDir(Command):
    def __init__(self) -> None:
        super().__init__()
        self.gen_dir = pydra.REQUIRED
        self.out = pydra.REQUIRED
        self.kind = pydra.REQUIRED
        self.layers = None

    def finalize(self) -> None:
        if self.kind not in ("prose", "tokens"):
            raise ValueError(f"kind must be prose or tokens, got {self.kind!r}")
        self.gen_dir = _path(self.gen_dir)
        self.out = _path(self.out)
        self.layers = _ints(self.layers)

    def execute(self) -> int:
        rep = convert_gen_dir(self.gen_dir, self.out, kind=self.kind, layers=self.layers)
        print(
            f"wrote {self.out}: kind={rep.kind} rows={rep.n_rows} layers={rep.layers} "
            f"empty={rep.n_empty} skipped={rep.skipped}"
        )
        return 0


class ConvertReadJson(Command):
    """The write-cell ``read.json`` of multi_concept_directed_modulation -> a contract file."""

    def __init__(self) -> None:
        super().__init__()
        self.read = pydra.REQUIRED
        self.out = pydra.REQUIRED

    def finalize(self) -> None:
        self.read = _path(self.read)
        self.out = _path(self.out)

    def execute(self) -> int:
        rep = convert_read_json(self.read, self.out)
        print(
            f"wrote {self.out}: kind={rep.kind} rows={rep.n_rows} layers={rep.layers} "
            f"empty={rep.n_empty} skipped={rep.skipped}"
        )
        return 0


def floor_columns() -> dict[str, dict[str, str]]:
    """The report's floor columns from the frozen baselines: lucky guessing (blind / described
    means) and prompt-only (judged rate), each only where the stamp matches the instrument."""

    def f(v: object) -> str:
        return "—" if v is None else f"{float(v):.3f}"

    lucky = {
        fam: f"{f(e.get('blind', {}).get('mean'))} / {f(e.get('described', {}).get('mean'))}"
        for fam, e in lucky_guessing.floors().items()
    }
    po = {}
    for fam, e in prompt_only.floors().items():
        note = " saturated" if fam in prompt_only.SATURATED else ""
        if e.get("higher_is_better") is False:
            note += " (lower is better)"
        elif e.get("metric") not in (None, "pass_rate"):
            note += f" ({e['metric']})"
        po[fam] = f(e.get("rate")) + note
    return {"lucky guess (blind / described)": lucky, "prompt-only": po}


class Baseline(Command):
    """Measure the lucky-guessing floor: a model shown only each family's option lists."""

    def __init__(self) -> None:
        super().__init__()
        self.families = "all"
        self.variant = "blind,described,uniform"
        self.draws = lucky_guessing.DEFAULT_DRAWS
        self.seed = lucky_guessing.DEFAULT_SEED
        self.limit = 0
        self.judge_model = ""
        self.concurrency = 64
        self.rpm = 240.0
        self.dry_run = False
        self.out = "outputs/baselines/lucky_guessing"

    def finalize(self) -> None:
        self.families = _strs(self.families) or ["all"]
        self.variant = _strs(self.variant) or []
        self.draws = _int(self.draws)
        self.seed = _int(self.seed)
        self.limit = _int(self.limit)
        self.judge_model = str(self.judge_model or "") or None
        self.concurrency = _int(self.concurrency)
        self.rpm = float(self.rpm)
        self.dry_run = _bool(self.dry_run)
        self.out = _path(self.out)

    def execute(self) -> int:
        names = list(lucky_guessing.BUILDERS) if self.families == ["all"] else self.families
        unknown = [f for f in names if f not in lucky_guessing.BUILDERS]
        bad = [v for v in self.variant if v not in lucky_guessing.VARIANTS]
        if unknown or bad or not self.variant or self.draws < 1:
            print(
                f"unknown families {unknown} / variants {bad}; known families "
                f"{sorted(lucky_guessing.BUILDERS)}, variants {lucky_guessing.VARIANTS}, "
                "draws >= 1",
                file=sys.stderr,
            )
            return EXIT_USAGE
        registry.load_all()
        try:
            judge = resolve(
                JudgeConfig(prompt_version=lucky_guessing.PROMPT_VERSION),
                flag=self.judge_model,
                env=os.environ,
            )
            for name in names:
                for variant in self.variant:
                    r = lucky_guessing.run_family(
                        name,
                        variant,
                        judge=judge,
                        out=self.out / name,
                        draws=self.draws,
                        seed=self.seed,
                        limit=self.limit,
                        concurrency=self.concurrency,
                        rpm=self.rpm,
                        dry_run=self.dry_run,
                    )
                    if not self.dry_run:
                        path = self.out / name / f"{variant}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(r, indent=1, ensure_ascii=False), "utf-8")
                    print(lucky_guessing.report_line(r))
        except JudgeConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return EXIT_JUDGE_CONFIG
        return 0


class Capable(Command):
    """Ask a model the banks' own questions and grade its answers: the gate a bank was built
    with, re-run on another model. A family whose rate drops has to be re-gated before its
    readouts mean anything on that model (AGENTS.md)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = pydra.REQUIRED
        self.families = "all"
        self.draws = capable_run.DEFAULT_DRAWS
        self.temperature = capable_run.DEFAULT_TEMPERATURE
        self.threshold = 0.0  # 0 = each family's own gate (10/10 for two of them)
        self.greedy = True
        self.reasoning_effort = capable_run.DEFAULT_REASONING_EFFORT
        self.limit = 0
        self.judge_model = ""
        self.concurrency = 64
        self.rpm = 240.0
        self.dry_run = False
        self.out = ""

    def finalize(self) -> None:
        self.model = str(self.model)
        self.families = _strs(self.families) or ["all"]
        self.draws = _int(self.draws)
        self.temperature = float(self.temperature)
        self.threshold = float(self.threshold) or None
        self.greedy = _bool(self.greedy)
        self.reasoning_effort = str(self.reasoning_effort or "") or None
        self.limit = _int(self.limit)
        self.judge_model = str(self.judge_model or "") or None
        self.concurrency = _int(self.concurrency)
        self.rpm = float(self.rpm)
        self.dry_run = _bool(self.dry_run)
        self.out = _path(self.out) if self.out else None

    def execute(self) -> int:
        names = sorted(capable_questions.BUILDERS) if self.families == ["all"] else self.families
        unknown = [f for f in names if f not in capable_questions.BUILDERS]
        if unknown:
            for f in unknown:
                why = capable_questions.NO_QUESTION.get(f, "unknown family")
                print(f"{f}: no capability question — {why}", file=sys.stderr)
            print(f"known: {sorted(capable_questions.BUILDERS)}", file=sys.stderr)
            return EXIT_USAGE
        out = self.out or Path("outputs/capable") / self.model.replace("/", "_")
        try:
            judge = resolve(
                JudgeConfig(prompt_version=capable_run.PROMPT_VERSION),
                flag=self.judge_model,
                env=os.environ,
            )
            for name in names:
                r = capable_run.run_family(
                    name,
                    model=self.model,
                    judge=judge,
                    out=out / name,
                    draws=self.draws,
                    temperature=self.temperature,
                    threshold=self.threshold,
                    greedy=self.greedy,
                    reasoning_effort=self.reasoning_effort,
                    limit=self.limit,
                    concurrency=self.concurrency,
                    rpm=self.rpm,
                    dry_run=self.dry_run,
                )
                if self.dry_run:
                    n = r["n_items"]
                    print(
                        f"{name}: {n} question{'' if n == 1 else 's'}, {r['n_calls']} answer "
                        "calls plus one grade per distinct answer; nothing sent"
                    )
                    continue
                path = out / name / "capable.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(r, indent=1, ensure_ascii=False), "utf-8")
                print(capable_run.report_line(r))
                if r.get("partial"):
                    print(f"  not covered: {r['partial']}")
                for key in ("subfamily", "leg", "src", "variant"):
                    split = capable_run.subfamily_rates(r, key)
                    if len(split) > 1:
                        print(f"  by {key}: " + "  ".join(f"{k}={v:.3f}" for k, v in split.items()))
                top = capable_run.answer_histogram(r)
                if top and top[0][1] > max(3, r["draws"]):
                    print("  most common answers: " + ", ".join(f"{t!r} x{n}" for t, n in top))
        except JudgeConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return EXIT_JUDGE_CONFIG
        return 0


class Plan(Command):
    """Write the read plan: for every item, what to render and how, which positions to read
    and at which layers (`docs/producing_readouts.md`). One JSONL per family under `out`."""

    def __init__(self) -> None:
        super().__init__()
        self.families = "all"
        self.out = "outputs/plan"
        self.limit = 0

    def finalize(self) -> None:
        self.families = _strs(self.families) or ["all"]
        self.out = _path(self.out)
        self.limit = _int(self.limit)

    def execute(self) -> int:
        names = readplan.families() if self.families == ["all"] else self.families
        unknown = [f for f in names if f not in readplan.families()]
        if unknown:
            print(f"unknown families {unknown}; known {readplan.families()}", file=sys.stderr)
            return EXIT_USAGE
        for name in names:
            specs = readplan.plan(name)[: self.limit or None]
            n = readplan.write(specs, self.out / f"{name}.jsonl")
            layers = sorted({L for s in specs for L in s.layers})
            kinds = sorted({s.positions["kind"] for s in specs})
            print(
                f"{name}: {n} items, render {sorted({s.render for s in specs})}, "
                f"positions {kinds}, layers {layers}"
            )
        return 0


class Produce(Command):
    """Produce readouts with the built-in producer (needs the `gpu` extra): a family into one
    readouts JSONL, or one prompt with `text=` over `positions=` and `layers=`."""

    def __init__(self) -> None:
        super().__init__()
        self.method = "logit_lens"
        self.model = "Qwen/Qwen3.6-27B"
        self.family = ""
        self.text = ""
        self.chat = False
        self.positions = "-1"  # an int, a comma list, or a rule kind such as all / final_token
        self.layers = None
        self.out = ""
        self.limit = 0
        self.items = None
        self.device = "cuda"

    def finalize(self) -> None:
        self.method = str(self.method)
        self.model = str(self.model)
        self.family = str(self.family or "")
        self.text = str(self.text or "")
        self.chat = _bool(self.chat)
        p = self.positions
        if isinstance(p, list | tuple):
            self.positions = [int(x) for x in p]
        elif str(p).lstrip("-").isdigit():
            self.positions = int(p)
        elif "," in str(p):
            self.positions = [int(x) for x in str(p).split(",")]
        else:
            self.positions = str(p)  # a rule kind such as all / final_token
        self.layers = _ints(self.layers)
        self.out = _path(self.out) if self.out else None
        self.limit = _int(self.limit)
        self.items = _strs(self.items)
        self.device = str(self.device)

    def execute(self) -> int:
        from wsbench.produce import METHODS, Producer, write

        if bool(self.family) == bool(self.text):
            print("give exactly one of family= or text=", file=sys.stderr)
            return EXIT_USAGE
        if self.method not in METHODS:
            print(f"unknown method {self.method!r}; known {sorted(METHODS)}", file=sys.stderr)
            return EXIT_USAGE
        if self.family and self.family not in readplan.families():
            print(f"unknown family {self.family!r}", file=sys.stderr)
            return EXIT_USAGE
        producer = Producer.load(self.model, self.method, device=self.device)
        if self.family:
            out = self.out or Path("outputs/readouts") / self.method / f"{self.family}.jsonl"
            path = producer.run_family(
                self.family, out, limit=self.limit, layers=self.layers, items=self.items
            )
            print(f"wrote {path}")
            return 0
        rows = producer.read_prompt(
            self.text, positions=self.positions, layers=self.layers, chat=self.chat
        )
        if self.out:
            print(f"wrote {write(rows, self.out)} ({len(rows)} rows)")
        else:
            for r in rows:
                print(json.dumps(r.contract(), ensure_ascii=False))
        return 0


class Freeze(Command):
    """Fold a finished baseline run into the tracked ``evals/baselines/<kind>.json``.
    ``kind=lucky_guessing``: ``src`` holds ``<family>/<variant>.json`` from ``wsbench baseline``.
    ``kind=prompt_only``: ``src`` holds ``<family>/results.json`` from ``wsbench run`` on the
    prompt-only readouts; ``source`` is the generation's ``run_config.json``."""

    def __init__(self) -> None:
        super().__init__()
        self.kind = "lucky_guessing"
        self.src = ""
        self.dst = ""
        self.source = ""

    def finalize(self) -> None:
        self.kind = str(self.kind)
        self.src = _path(self.src) if self.src else None
        self.dst = _path(self.dst) if self.dst else None
        self.source = _path(self.source) if self.source else None

    def execute(self) -> int:
        try:
            if self.kind == "lucky_guessing":
                dst = self.dst or lucky_guessing.FROZEN
                src = self.src or Path("outputs/baselines/lucky_guessing")
                frozen = lucky_guessing.freeze(src, dst)
            elif self.kind == "prompt_only":
                dst = self.dst or prompt_only.FROZEN
                src = self.src or Path("outputs/prompt-only")
                frozen = prompt_only.freeze(src, dst, source=self.source)
            else:
                print(
                    f"unknown baseline kind {self.kind!r}; known: lucky_guessing, prompt_only",
                    file=sys.stderr,
                )
                return EXIT_USAGE
        except (ValueError, OSError) as e:
            print(str(e), file=sys.stderr)
            return EXIT_USAGE
        print(f"froze {sorted(k for k in frozen if not k.startswith('_'))} -> {dst}")
        return 0


COMMANDS: dict[str, type[Command]] = {
    "list": ListFamilies,
    "judge": JudgeFamily,
    "run": RunFamilies,
    "report": ReportRuns,
    "baseline": Baseline,
    "capable": Capable,
    "plan": Plan,
    "produce": Produce,
    "freeze": Freeze,
    "convert-gen-dir": ConvertGenDir,
    "convert-read-json": ConvertReadJson,
}


def _n_items(spec: EvalSpec) -> str:
    path = registry.REPO_ROOT / spec.bank
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "?"
    if isinstance(d, list):
        return str(len(d))
    if isinstance(d, dict):
        for key in ("items", "prompts"):  # {meta, items} banks; the jlens acts manifest
            if isinstance(d.get(key), list):
                return str(len(d[key]))
    return "?"


def _unknown(name: str) -> int:
    known = ", ".join(sorted(registry.FAMILIES)) or "(none)"
    print(f"unknown family {name!r}; known families: {known}", file=sys.stderr)
    return EXIT_USAGE


def _write_summary(
    out: Path, results: list[FamilyResult], notes: list[FamilyOutcome] | None = None
) -> str:
    text = markdown_table(results, macro(results))
    if notes:
        text += "\n## skipped / failed\n"
        text += "".join(f"- {o.family}: {o.status} — {o.error}\n" for o in notes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def _usage() -> str:
    return "usage: wsbench <" + "|".join(COMMANDS) + "> [key=value ...] [--show | --help]"


def _defaults(command: Command) -> str:
    keys = "  ".join(
        f"{k}=<required>" if v is pydra.REQUIRED else f"{k}={v!r}"
        for k, v in command.to_dict().items()
    )
    return f"{_usage()}\nkeys: {keys}"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--help", "-h"):
        print(_usage())
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(_usage(), file=sys.stderr)
        return EXIT_USAGE
    command = COMMANDS[argv[0]]()
    if any(a in ("--help", "-h") for a in argv[1:]):
        print(_defaults(command))
        return 0
    try:
        show = pydra.apply_overrides(command, argv[1:])
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        print(f"{argv[0]}: {e}\n{_usage()}", file=sys.stderr)
        return EXIT_USAGE
    if show:
        print(json.dumps(command.to_dict(), indent=1))
        return 0
    registry.load_all()
    return command.execute()


if __name__ == "__main__":
    sys.exit(main())
