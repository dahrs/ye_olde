"""Resolves (lang_code, year) to weighted corpus partitions — spec §2, §3a.

Today's actual job (spec §9's open decision on λ): choosing the temporal-
decay rate `retrieval/` reranks results with — `score = similarity *
exp(-λ * year_gap)` (spec §4). Not a fixed global constant, and not spec's
original per-language-code nominal-range "boundary blend" design either —
neither can be validated against real data yet (only `enm`/`eng` exist, no
`ang`; only one work exists per language pair), so both would be tuned
against nothing.

Instead, λ is adaptive: gentle when little data exists near the queried
year (don't punish distance when there's nothing closer to use), harsher
when a lot does (can afford to be strict when genuinely nearby data
exists). Deliberately keyed on *local* density — how much exists near this
specific query — not a language's total document count: a language with
abundant data everywhere except near this particular year should still get
a gentle λ for this query. `local_count` is supplied by `retrieval/` from
candidates it already fetched for the same query being reranked — this
module doesn't fetch or filter anything itself, and the count it's given is
never used to limit what gets queried, only to decide how harsh λ should be.
"""

from __future__ import annotations

import math

# How close (in years) a candidate has to be to the queried year to count
# toward "local" density. Open decision (spec §9), no empirical grounding
# yet — chosen to be wide enough that Sir Gawayne's own 599-year enm/eng
# gap reads as sparse (0 local candidates on either side), narrow enough
# that "nearby" still means something once denser periods exist.
LOCAL_WINDOW_YEARS = 100

# Asymptotic ceiling for λ — how harsh reranking gets once local data is
# genuinely abundant. Same caveat as above: an open decision, not tuned
# against real query examples yet.
LAMBDA_MAX = 0.01

# Controls how fast λ ramps up toward LAMBDA_MAX as local_count grows.
# local_count == _RAMP_K -> ~63% of LAMBDA_MAX (one time-constant into the
# curve); local_count == 1000 -> ~96% of LAMBDA_MAX with this value, close
# to but not exactly at the ceiling, deliberately avoiding a hard cliff at
# any single count.
_RAMP_K = 300.0


def compute_lambda(local_count: int, *, lambda_max: float = LAMBDA_MAX, k: float = _RAMP_K) -> float:
    """Adaptive temporal-decay rate for retrieval/'s reranking formula.

    `local_count` is how many candidate passages were already found within
    `LOCAL_WINDOW_YEARS` of the queried year — supplied by the caller
    (`retrieval/`), not looked up here:
      - `local_count == 0` -> `λ == 0`: no local data at all, so don't
        punish distance — use whatever exists, however far away.
      - `local_count` large -> `λ` approaches `lambda_max`: plenty of local
        data, so it's safe to favor genuinely nearby results more strongly.
    A negative count (shouldn't happen, but not this function's job to
    validate the caller's arithmetic) is treated the same as zero.
    """
    if local_count <= 0:
        return 0.0
    return lambda_max * (1.0 - math.exp(-local_count / k))
