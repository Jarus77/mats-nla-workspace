from wsbench.judge_config import (
    DEFAULT_JUDGE,
    DEFAULT_REASONING,
    ENV_OVERRIDE,
    JudgeConfig,
    resolve,
)


def test_family_default():
    cfg = JudgeConfig()
    r = resolve(cfg, env={})
    assert r.model == DEFAULT_JUDGE
    assert r.source == "family"
    assert r.pinned is True
    assert r.reasoning == DEFAULT_REASONING


def test_precedence_flag_over_env_over_family():
    cfg = JudgeConfig(model="claude-sonnet-5", reasoning=None)
    r = resolve(cfg, flag="google/gemini-3.8-flash", env={ENV_OVERRIDE: "x/y"})
    assert r.model == "google/gemini-3.8-flash" and r.source == "flag" and not r.pinned
    r = resolve(cfg, env={ENV_OVERRIDE: "x/y"})
    assert r.model == "x/y" and r.source == "env" and not r.pinned
    r = resolve(cfg, env={ENV_OVERRIDE: ""})
    assert r.model == "claude-sonnet-5" and r.source == "family" and r.pinned


def test_override_equal_to_pin_is_pinned():
    cfg = JudgeConfig(model="claude-sonnet-5")
    r = resolve(cfg, flag="claude-sonnet-5", env={ENV_OVERRIDE: "other"})
    assert r.pinned is True and r.source == "flag"
    r = resolve(cfg, env={ENV_OVERRIDE: "claude-sonnet-5"})
    assert r.pinned is True and r.source == "env"


def test_reasoning_defaults_per_route():
    assert resolve(JudgeConfig(model="claude-sonnet-5"), env={}).reasoning is None
    assert (
        resolve(JudgeConfig(model="google/gemini-3.8-flash"), env={}).reasoning == DEFAULT_REASONING
    )
    # explicit reasoning wins on either route
    assert resolve(
        JudgeConfig(model="claude-sonnet-5", reasoning={"effort": "high"}), env={}
    ).reasoning == {"effort": "high"}
    # an override that changes the route re-derives the default
    r = resolve(JudgeConfig(model="claude-sonnet-5"), flag="google/gemini-3.8-flash", env={})
    assert r.reasoning == DEFAULT_REASONING
    r = resolve(JudgeConfig(model="google/gemini-3.8-flash"), flag="claude-sonnet-5", env={})
    assert r.reasoning is None


def test_aux_models_default_empty():
    assert dict(JudgeConfig().aux_models) == {}
    assert JudgeConfig().prompt_version == "v1"
