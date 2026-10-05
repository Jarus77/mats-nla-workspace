"""The prompts this family sends: the forced-choice judge shared by the multi-token families."""

from wsbench.multitoken.prompts import PROMPT_VERSION, PROMPTS, SCHEMA, SYSTEM, render_user

__all__ = ["PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]
