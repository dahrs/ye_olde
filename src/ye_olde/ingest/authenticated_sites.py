"""The small, separate path for a site that needs identification before it'll
serve content -- kept deliberately apart from `acquire.fetch_source`'s plain
path (Gutenberg and "others"), which stays exactly as it is and needs none of
this.

First and so far only real case: Gallica (gallica.bnf.fr). Confirmed live:
its whole domain sits behind Cloudflare *and* a second ALTCHA proof-of-work
check. Cloudflare's own check resolves automatically after a 2-5 minute
wait; ALTCHA's "I'm not a robot" checkbox needs a real browser actually
running JavaScript to solve -- no plain HTTP request, however honestly
identified or throttled, gets past it (tried: an honest, non-browser-
spoofing, single slow request still got the challenge page). What *does*
work, confirmed live by actually doing it: solving it once in a real browser
grants a session, cookie-based, good for 24 hours. That's the legitimate
pattern this module supports -- a human unlocks access, the tool reuses that
access for the rest of the day -- never an attempt to automate solving
ALTCHA itself, which would be exactly the bot-check circumvention this
project has already correctly refused to attempt once.

This registry is deliberately not Gallica-specific in shape, even though
Gallica is the only entry today -- any other site that turns out to need the
same "manually-supplied cookie, a particular User-Agent, a mandatory minimum
interval between requests" treatment is one more `_REGISTRY` entry, not a
new code path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

from ..config import Settings, get_settings


@dataclass(frozen=True)
class _SiteAuth:
    cookie_setting: str  # the `Settings` attribute name holding this site's cookie
    user_agent: str
    min_interval_seconds: float


_REGISTRY: dict[str, _SiteAuth] = {
    "gallica.bnf.fr": _SiteAuth(
        cookie_setting="gallica_cookie",
        user_agent="ye_olde-corpus-acquirer/0.1 (+personal research corpus tool; see project README)",
        # Gallica's own documented per-minute caps differ by content type (texteBrut 5/min, PDF
        # 4/min, full IIIF images 5/min) -- rather than track each separately for this project's
        # modest scale, one global interval safely under the *strictest* of them (PDF, 4/min ->
        # 15s) keeps every request type within its own limit without needing per-type bucketing.
        min_interval_seconds=16.0,
    ),
}

# Process-lifetime only, keyed by domain -- when each registered domain was last actually
# fetched, so `headers_for` can throttle the *next* call to it regardless of which caller
# (acquire.py's real download, or survey.py's crawl) makes it.
_last_call_at: dict[str, float] = {}


def headers_for(url: str, *, settings: Settings | None = None) -> dict[str, str]:
    """`{}` for any URL whose domain isn't in `_REGISTRY` -- the default,
    unchanged case every existing caller already gets. For a registered
    domain: sleeps first if needed to respect that site's own minimum
    interval since the last call to it, then returns the `Cookie`/
    `User-Agent` headers to merge into the actual request
    (`acquire.fetch_source`'s new `extra_headers` parameter).

    Raises `ValueError` -- naming the exact `.env` variable to set -- if a
    registered domain's cookie setting is empty, rather than silently
    sending an unauthenticated request to a site already confirmed to wall
    those off (CLAUDE.md "Error handling": this is a plain invalid/missing
    configuration, exactly what `ValueError` is for, not a `YeOldeError`
    subclass -- nothing downstream needs to distinguish this from any other
    bad-input case).
    """
    domain = urlparse(url).netloc
    site = _REGISTRY.get(domain)
    if site is None:
        return {}

    settings = settings or get_settings()
    cookie = getattr(settings, site.cookie_setting)
    if not cookie:
        env_var = site.cookie_setting.upper()
        raise ValueError(
            f"{domain} requires a manually-obtained session cookie, but {env_var} isn't set in "
            f".env -- see .env.example's comment on {env_var} for how to get one."
        )

    last_call = _last_call_at.get(domain)
    if last_call is not None:
        wait = site.min_interval_seconds - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
    _last_call_at[domain] = time.monotonic()

    return {"Cookie": cookie, "User-Agent": site.user_agent}
