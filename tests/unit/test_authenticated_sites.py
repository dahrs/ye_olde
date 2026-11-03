"""Unit tests for authenticated_sites.py: the registry lookup, the clear
error when a registered site's cookie isn't configured, and the per-domain
throttle -- no real network or real waiting (`time.sleep`/`time.monotonic`
mocked throughout).
"""

from __future__ import annotations

import pytest

from ye_olde.config import Settings
from ye_olde.ingest import authenticated_sites


@pytest.fixture(autouse=True)
def _clear_throttle_state():
    authenticated_sites._last_call_at.clear()
    yield
    authenticated_sites._last_call_at.clear()


def test_headers_for_unregistered_domain_returns_empty_and_does_not_throttle(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(authenticated_sites.time, "sleep", lambda s: sleep_calls.append(s))
    assert authenticated_sites.headers_for("https://www.gutenberg.org/ebooks/111") == {}
    assert sleep_calls == []


def test_headers_for_registered_domain_raises_when_cookie_unset():
    settings = Settings(gallica_cookie="")
    with pytest.raises(ValueError, match="GALLICA_COOKIE"):
        authenticated_sites.headers_for("https://gallica.bnf.fr/ark:/12148/x.texteBrut", settings=settings)


def test_headers_for_registered_domain_returns_cookie_and_user_agent(monkeypatch):
    monkeypatch.setattr(authenticated_sites.time, "sleep", lambda s: None)
    settings = Settings(gallica_cookie="cf_clearance=abc123; altcha=xyz")
    headers = authenticated_sites.headers_for("https://gallica.bnf.fr/ark:/12148/x.texteBrut", settings=settings)
    assert headers["Cookie"] == "cf_clearance=abc123; altcha=xyz"
    assert "User-Agent" in headers


def test_headers_for_throttles_consecutive_calls_to_the_same_registered_domain(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(authenticated_sites.time, "monotonic", lambda: now[0])
    sleep_calls = []
    monkeypatch.setattr(authenticated_sites.time, "sleep", lambda s: sleep_calls.append(s))
    settings = Settings(gallica_cookie="cf_clearance=abc123")

    authenticated_sites.headers_for("https://gallica.bnf.fr/ark:/12148/x.texteBrut", settings=settings)
    assert sleep_calls == []  # first call to this domain -- nothing to wait for yet

    now[0] += 1.0  # only 1s later, well under the ~16s minimum interval
    authenticated_sites.headers_for("https://gallica.bnf.fr/ark:/12148/y.texteBrut", settings=settings)
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0


def test_headers_for_a_different_domains_throttle_is_independent(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(authenticated_sites.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(authenticated_sites.time, "sleep", lambda s: None)
    monkeypatch.setitem(
        authenticated_sites._REGISTRY,
        "example.com",
        authenticated_sites._SiteAuth(cookie_setting="gallica_cookie", user_agent="test-ua", min_interval_seconds=16.0),
    )
    settings = Settings(gallica_cookie="cf_clearance=abc123")

    authenticated_sites.headers_for("https://gallica.bnf.fr/ark:/12148/x.texteBrut", settings=settings)
    sleep_calls = []
    monkeypatch.setattr(authenticated_sites.time, "sleep", lambda s: sleep_calls.append(s))
    # A different registered domain, visited for the first time -- its own throttle state
    # is independent of gallica.bnf.fr's, so this must not wait at all.
    authenticated_sites.headers_for("https://example.com/doc", settings=settings)
    assert sleep_calls == []
