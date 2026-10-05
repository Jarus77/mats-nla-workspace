"""Poetry: a single-token basic family on the shared bank judge."""

from wsbench.basic.family import bank_family
from wsbench.registry import register

SPEC = register(bank_family("poetry", "Poetry"))
