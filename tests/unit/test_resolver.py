"""Unit tests for the adaptive temporal-decay rate (spec §9's open decision
on λ). `compute_lambda` takes a local candidate count already found by
`retrieval/` — it doesn't fetch or filter anything itself.
"""

from __future__ import annotations

import math

import pytest

from ye_olde import resolver


def test_compute_lambda_zero_local_count_is_fully_gentle():
    assert resolver.compute_lambda(0) == 0.0


def test_compute_lambda_negative_count_treated_as_zero():
    assert resolver.compute_lambda(-5) == 0.0


def test_compute_lambda_increases_strictly_with_local_count():
    values = [resolver.compute_lambda(n) for n in (0, 10, 100, 300, 1000, 10_000)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_compute_lambda_approaches_but_never_reaches_lambda_max():
    # Not too large a count: past roughly 12_000 here, `1 - exp(-n/k)` rounds
    # to exactly 1.0 in float64 (the subtracted term drops below float64's
    # precision relative to 1.0, well before exp() itself would underflow to
    # zero), which would make this a test of floating-point rounding instead
    # of the actual "approaches but never reaches" behavior being checked.
    lam = resolver.compute_lambda(1_000)
    assert lam < resolver.LAMBDA_MAX
    assert lam > 0.9 * resolver.LAMBDA_MAX


def test_compute_lambda_at_ramp_k_is_one_time_constant_into_the_curve():
    # 1 - exp(-1) is the standard "63%" point for this family of saturating
    # curves -- a sanity check that the ramp shape is what it's meant to be.
    lam = resolver.compute_lambda(int(resolver._RAMP_K))
    expected = resolver.LAMBDA_MAX * (1 - math.exp(-1))
    assert lam == pytest.approx(expected, rel=1e-6)


def test_compute_lambda_respects_custom_lambda_max_and_k():
    lam = resolver.compute_lambda(50, lambda_max=1.0, k=50.0)
    assert lam == pytest.approx(1.0 - math.exp(-1), rel=1e-6)
