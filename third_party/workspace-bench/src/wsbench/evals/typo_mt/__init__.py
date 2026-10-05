"""Typo (multi-token): a multi-token basic family on the forced-choice judge."""

from wsbench.multitoken.family import mt_family
from wsbench.registry import register

SPEC = register(mt_family("typo_mt", "Typo (multi-token)", calls_per_arm="≈ 1.1k"))
