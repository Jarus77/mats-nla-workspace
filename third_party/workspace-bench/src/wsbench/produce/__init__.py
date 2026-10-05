"""Produce readouts from a model, a method and a prompt, in the benchmark's contract; needs the
``gpu`` extra. The tour is ``docs/producing_readouts.md``."""

from .backend import DEFAULT_MODEL, Backend
from .methods import METHODS, NLA, JLens, LogitLens, Method, OLens, Readout, RLens, Sampling, method
from .producer import Producer, Row, write
from .render import Rendered, render, render_text

__all__ = [
    "DEFAULT_MODEL",
    "METHODS",
    "NLA",
    "Backend",
    "JLens",
    "LogitLens",
    "Method",
    "OLens",
    "Producer",
    "RLens",
    "Readout",
    "Rendered",
    "Row",
    "Sampling",
    "method",
    "render",
    "render_text",
    "write",
]
