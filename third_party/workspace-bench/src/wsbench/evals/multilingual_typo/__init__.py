"""Multilingual typo: a multi-token basic family on the forced-choice judge."""

from wsbench.multitoken.family import mt_family
from wsbench.registry import register

SPEC = register(mt_family("multilingual_typo", "Multilingual typo", calls_per_arm="≈ 2.2k"))
