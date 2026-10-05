"""Download the sprint's weights into the `mats-weights` volume (HF cache layout under /weights/hf).

    modal run --detach infra/download_weights.py              # all repos, one container each
    modal run --detach infra/download_weights.py --only qwen  # a subset by key (comma-separated)

Each repo runs as its own function call (`map(..., return_exceptions=True)`), so one failure
does not kill the rest; each call commits the volume when it finishes. snapshot_download
resumes, so re-running after a failure only fetches what is missing.
"""

import time
from dataclasses import asdict, dataclass

import modal

from modal_common import (
    HF_HOME_DIR,
    HF_SECRETS,
    RESULTS_DIR,
    VOLUMES,
    WEIGHTS_DIR,
    download_image,
    weights_volume,
)

app = modal.App("mats-download-weights")


@dataclass(frozen=True)
class RepoSpec:
    key: str
    repo_id: str
    repo_type: str = "model"
    allow_patterns: tuple[str, ...] | None = None


REPOS: tuple[RepoSpec, ...] = (
    RepoSpec("qwen", "Qwen/Qwen3.6-27B"),  # ~55.6 GB
    RepoSpec(
        "nla",
        "ceselder/qwen3.6-27b-nla-rl",  # ~93 GB selected of ~402 GB
        allow_patterns=(
            "av_base/*",
            "av_rl_adapters/iter_000400/*",
            "av_rl_adapters/iter_000300/*",
            "ar_reconstructor/*",
            "nla_meta.yaml",
            "run_config.yaml",
            "data/*",
            "README.md",
        ),
    ),
    RepoSpec("olens_ar", "agu18dec/olens_and_ar"),  # ~1 GB
    RepoSpec(
        "olens_default",
        "agu18dec/local-workspace",  # WorkspaceBench default olens LoRA, ~0.9 GB
        repo_type="dataset",
        allow_patterns=("ckpts/ao/rl/s3d.ddp600.s0/iter_000600/*",),
    ),
    RepoSpec(
        "lenses",
        "camilablank/workspace-lenses",  # j-lens, r-lens, template-lens etc., ~25 GB
        allow_patterns=("qwen3.6-27b/**",),
    ),
)


def _dir_size_gb(path: str) -> float:
    import os

    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            # snapshot entries are symlinks into blobs/; getsize follows them
            total += os.path.getsize(os.path.join(root, name))
    return total / 1e9


@app.function(
    image=download_image,
    volumes=VOLUMES,
    secrets=HF_SECRETS,
    cpu=8,
    memory=16384,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
)
def download_repo(spec: dict) -> dict:
    from huggingface_hub import snapshot_download

    s = RepoSpec(**{**spec, "allow_patterns": tuple(spec["allow_patterns"] or ()) or None})
    t0 = time.time()
    print(f"[{s.key}] start {s.repo_type}:{s.repo_id} patterns={s.allow_patterns}", flush=True)
    path = snapshot_download(
        repo_id=s.repo_id,
        repo_type=s.repo_type,
        allow_patterns=list(s.allow_patterns) if s.allow_patterns else None,
        max_workers=16,
    )
    weights_volume.commit()
    size = _dir_size_gb(path)
    mins = (time.time() - t0) / 60
    print(f"[{s.key}] done -> {path} ({size:.2f} GB, {mins:.1f} min); volume committed", flush=True)
    return {"key": s.key, "repo_id": s.repo_id, "path": path, "gb": round(size, 2), "min": round(mins, 1)}


@app.local_entrypoint()
def main(only: str = "") -> None:
    wanted = {k.strip() for k in only.split(",") if k.strip()}
    specs = [asdict(s) for s in REPOS if not wanted or s.key in wanted]
    print(f"HF cache: {HF_HOME_DIR} on volume mounted at {WEIGHTS_DIR}; results vol at {RESULTS_DIR}")
    print("launching:", [s["key"] for s in specs])
    failures = 0
    for spec, result in zip(specs, download_repo.map(specs, return_exceptions=True)):
        if isinstance(result, BaseException):
            failures += 1
            print(f"FAILED {spec['key']} ({spec['repo_id']}): {type(result).__name__}: {result}")
        else:
            print(f"OK {result}")
    print(f"finished: {len(specs) - failures} ok, {failures} failed")
