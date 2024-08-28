#!/usr/bin/env python3

from .bm1 import M, α, β, κ, ρ


def c2y(c):
    # Convert composition field to order parameter
    return (2 * c - α - β) / (β - α)


def y2c(y):
    # Convert order parameter field to composition
    return 0.5 * (β - α) * (1 + y) + α


def gamma():
    # Compute gradient energy coefficient
    return κ / (ρ * (β - α) ** 2)


def t2τ(t):
    # Nondimensionalize time
    return t / (ρ * M * (β - α) ** 2)


def τ2t(τ):
    # Dimensionalize time
    return (ρ * M * (β - α) ** 2) * τ
