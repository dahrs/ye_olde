"""Unit tests for survey.py: link extraction (Gutenberg-shaped and
deliberately not), the two optional Gutenberg shortcuts, batched
classification, and the crawl's frontier/depth-limit/checkpoint behavior
-- all against mocked `fetch_source`/`call_llm_json`, no real network.
"""

from __future__ import annotations

from ye_olde.common.errors import LLMEmptyResponseError
from ye_olde.ingest import survey
from ye_olde.ingest.acquire import FetchedContent
from ye_olde.ingest.checkpoint import Checkpoint
from ye_olde.ingest.survey import PageLink

_GUTENBERG_SHAPED_HTML = """
<html><body>
<ul>
<li class="pgdbetext"><a href="/ebooks/111">A Grammar Work</a> (German) (as Author)</li>
<li class="pgdbetext"><a href="/ebooks/111">A Grammar Work</a> (Old English) (as Author)</li>
<li class="pgdbetext"><a href="/ebooks/222">A Poem</a> (English) (as Translator)</li>
<li class="pgdbetext"><a href="/ebooks/222">A Poem</a> (Old English) (as Translator)</li>
</ul>
</body></html>
"""

_NON_GUTENBERG_HTML = """
<html><body>
<table>
<tr><td><a href="/works/42">Some Old Manuscript</a></td><td>13th century, Occitan</td></tr>
</table>
</body></html>
"""

_BIBREC_HTML = """
<html><head><title>A Grammar Work</title></head><body>
<table id="bibrec">
<tr><th>Author</th><td>Someone Ancient, 955?-1020?</td></tr>
<tr><th>Editor</th><td>Someone Modern, 1844-1895</td></tr>
<tr><th>Language</th><td>German</td></tr>
<tr><th>Language</th><td>Old English</td></tr>
<tr><th>Subject</th><td>Old English -- Grammar</td></tr>
</table>
<details class="other-formats">
<div class="other-format-row">
<a class="other-format-link" href="/ebooks/111.txt.utf-8" type="text/plain; charset=utf-8">Plain Text</a>
</div>
<div class="other-format-row">
<a class="other-format-link" href="/ebooks/111.epub.images" type="application/epub+zip">EPUB</a>
</div>
</details>
</body></html>
"""


def test_extract_links_handles_gutenberg_shaped_markup():
    links = survey.extract_links(_GUTENBERG_SHAPED_HTML, "https://www.gutenberg.org/browse/languages/ang")
    assert len(links) == 2  # deduped by absolute URL, not by (url, language) row
    assert links[0].url == "https://www.gutenberg.org/ebooks/111"
    assert "German" in links[0].context and "Author" in links[0].context


def test_extract_links_handles_non_gutenberg_markup():
    links = survey.extract_links(_NON_GUTENBERG_HTML, "https://gallica.example/index")
    assert len(links) == 1
    assert links[0].url == "https://gallica.example/works/42"
    assert "13th century" in links[0].context


def test_extract_links_skips_fragment_and_javascript_links():
    html = '<a href="#top">top</a><a href="javascript:void(0)">nope</a><a href="/real">real</a>'
    links = survey.extract_links(html, "https://example.com")
    assert [link.url for link in links] == ["https://example.com/real"]


def test_extract_links_strips_nav_header_footer_chrome():
    # Confirmed live against the real Gutenberg pages: their shared site navigation
    # (donate/help/about, present on every page) was consuming entire classification
    # batches before any real per-page content was reached -- this must never reach
    # extract_links' output at all, not just be classified "irrelevant" downstream.
    html = """
    <header><a href="/donate">Donate</a></header>
    <nav><a href="/help">Help</a></nav>
    <body><a href="/ebooks/1">A Real Book</a></body>
    <footer><a href="/about">About</a></footer>
    """
    links = survey.extract_links(html, "https://example.com")
    assert [link.url for link in links] == ["https://example.com/ebooks/1"]


def test_extract_links_drops_a_link_with_no_text_and_no_context():
    # Confirmed live: sending a link like this to the LLM is what caused it to reply
    # with one fewer decision than links given, silently shifting every later
    # decision's alignment by one position -- must never reach the LLM at all.
    html = '<a href="/skip"></a><a href="/real">A Real Book</a>'
    links = survey.extract_links(html, "https://example.com")
    assert [link.url for link in links] == ["https://example.com/real"]


def test_extract_links_keeps_a_link_with_context_but_no_own_text():
    # An image-only link still carries a signal via its surrounding block -- only a
    # link with NEITHER its own text NOR usable context should be dropped.
    html = '<tr><td><a href="/icon-only"><img src="x.png"></a> an old manuscript, 12th century</td></tr>'
    links = survey.extract_links(html, "https://example.com")
    assert len(links) == 1
    assert "12th century" in links[0].context


def test_url_suffix_is_a_document():
    assert survey._url_suffix_is_a_document("https://example.com/book.txt") is True
    assert survey._url_suffix_is_a_document("https://example.com/book.pdf") is True
    assert survey._url_suffix_is_a_document("https://example.com/ebooks/111") is False


def test_is_gutenberg_author_listing():
    assert survey._is_gutenberg_author_listing("https://www.gutenberg.org/browse/authors/u#a49") is True
    assert survey._is_gutenberg_author_listing("https://www.gutenberg.org/ebooks/111") is False
    assert survey._is_gutenberg_author_listing("https://gallica.example/authors/u") is False


def test_gutenberg_format_links_finds_plain_text_only():
    links = survey._gutenberg_format_links(_BIBREC_HTML)
    assert links == ["https://www.gutenberg.org/ebooks/111.txt.utf-8"]


def test_gutenberg_format_links_empty_for_non_gutenberg_page():
    assert survey._gutenberg_format_links(_NON_GUTENBERG_HTML) == []


def test_gutenberg_bibrec_hint_renders_known_fields():
    hint = survey._gutenberg_bibrec_hint(_BIBREC_HTML)
    assert hint is not None
    assert "Someone Ancient, 955?-1020?" in hint
    assert "Old English -- Grammar" in hint


def test_gutenberg_bibrec_hint_none_without_bibrec_table():
    assert survey._gutenberg_bibrec_hint(_NON_GUTENBERG_HTML) is None


def test_classify_links_parses_batched_reply(monkeypatch):
    links = [
        PageLink(url="https://x/1", text="A", context="ctx1"),
        PageLink(url="https://x/2", text="B", context="ctx2"),
    ]
    reply = [
        {"decision": "candidate", "category": "old_text_candidate", "reasoning": "r1", "confidence": 0.8},
        {"decision": "irrelevant", "reasoning": "r2", "confidence": 0.9},
    ]
    monkeypatch.setattr(survey, "call_llm_json", lambda *a, **k: reply)
    decisions = survey.classify_links(links, batch_size=10)
    assert decisions[0].decision == "candidate"
    assert decisions[0].category == "old_text_candidate"
    assert decisions[1].decision == "irrelevant"


def test_classify_links_unwraps_a_bare_object_for_a_single_link_batch(monkeypatch):
    # Confirmed live: asked to classify a batch of exactly one link, a real model
    # sometimes replies with a bare JSON object instead of a one-element array, even
    # though the prompt asks for an array -- this is _visit_page's Gutenberg-format-link
    # path's *common* case (always exactly one pseudo-link), not a rare edge case.
    link = [PageLink(url="https://x/1", text="A", context="ctx1")]
    bare_reply = {"decision": "candidate", "category": "old_text_candidate", "reasoning": "r1", "confidence": 0.9}
    monkeypatch.setattr(survey, "call_llm_json", lambda *a, **k: bare_reply)
    decisions = survey.classify_links(link)
    assert len(decisions) == 1
    assert decisions[0].decision == "candidate"
    assert decisions[0].category == "old_text_candidate"


def test_classify_links_recovers_via_one_repair_retry_on_count_mismatch(monkeypatch):
    # Confirmed live: 22 of 1418 calls in one real run undercounted by one, even at batch
    # sizes > 1 (not just the single-item-dict case) -- most plausibly the model treating
    # two similarly-worded links as one. One repair retry, showing the model its own bad
    # reply and the exact count needed, recovered this in testing.
    links = [
        PageLink(url="https://x/1", text="A", context="ctx1"),
        PageLink(url="https://x/2", text="B", context="ctx2"),
    ]
    calls = []

    def fake_call(prompt, **k):
        calls.append(prompt)
        if len(calls) == 1:
            return [{"decision": "candidate", "category": "old_text_candidate", "reasoning": "merged"}]  # undercounted
        return [
            {"decision": "candidate", "category": "old_text_candidate", "reasoning": "r1"},
            {"decision": "irrelevant", "reasoning": "r2"},
        ]

    monkeypatch.setattr(survey, "call_llm_json", fake_call)
    decisions = survey.classify_links(links)
    assert len(calls) == 2  # the repair retry actually happened
    assert len(decisions) == 2
    assert decisions[1].decision == "irrelevant"


def test_classify_links_degrades_a_batch_to_uncertain_if_repair_retry_also_fails(monkeypatch):
    # Confirmed live at real scale: raising here would discard every OTHER batch's already-
    # correct classifications in the same classify_links call too (a 146-link page needing
    # ~10 batches lost all of them over exactly one unrecoverable batch). Degrading just the
    # broken batch, not the whole call, is the fix.
    links = [PageLink(url="https://x/1", text="A", context="ctx1")]
    calls = []
    monkeypatch.setattr(survey, "call_llm_json", lambda *a, **k: (calls.append(1), [])[1])
    decisions = survey.classify_links(links)
    assert len(calls) == 2  # one initial call + one repair retry, not retried indefinitely
    assert len(decisions) == 1
    assert decisions[0].decision == "uncertain"


def test_classify_links_other_batches_survive_one_batch_degrading(monkeypatch):
    links = [
        PageLink(url="https://x/1", text="A", context="ctx1"),
        PageLink(url="https://x/2", text="B", context="ctx2"),
    ]
    calls = []

    def fake_call(prompt, **k):
        calls.append(prompt)
        if len(calls) <= 2:
            return []  # batch 1 (both initial call and its repair retry) permanently broken
        return [{"decision": "candidate", "category": "old_text_candidate", "reasoning": "r"}]

    monkeypatch.setattr(survey, "call_llm_json", fake_call)
    decisions = survey.classify_links(links, batch_size=1)
    assert decisions[0].decision == "uncertain"  # batch 1 degraded
    assert decisions[1].decision == "candidate"  # batch 2 still succeeded normally


def _no_sleep(monkeypatch):
    monkeypatch.setattr(survey.time, "sleep", lambda *_: None)


def test_crawl_resolves_candidate_straight_to_a_document(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        survey, "fetch_source", lambda url: FetchedContent(raw=b"irrelevant", content_type="text/plain", final_url=url)
    )
    results = survey.crawl("https://example.com/book.txt")
    assert len(results) == 1
    assert results[0].document_url == "https://example.com/book.txt"


def test_crawl_follows_a_link_then_finds_gutenberg_format_link(monkeypatch):
    _no_sleep(monkeypatch)

    def fake_fetch(url):
        if url == "https://www.gutenberg.org/browse/languages/ang":
            return FetchedContent(raw=_GUTENBERG_SHAPED_HTML.encode(), content_type="text/html", final_url=url)
        if url == "https://www.gutenberg.org/ebooks/111":
            return FetchedContent(raw=_BIBREC_HTML.encode(), content_type="text/html", final_url=url)
        raise AssertionError(f"unexpected fetch: {url}")

    def fake_classify(links, *, model=None, batch_size=15):
        # The index page's two links: follow the grammar work, skip the poem.
        if len(links) == 2:
            return [
                survey.LinkDecision(decision="follow", reasoning="looks like an old grammar"),
                survey.LinkDecision(decision="irrelevant", reasoning="not relevant for this test"),
            ]
        # The pseudo-link built from the resolved bibrec+format-link page.
        return [survey.LinkDecision(decision="candidate", category="bilingual_reference", reasoning="grammar work")]

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(survey, "classify_links", fake_classify)

    results = survey.crawl("https://www.gutenberg.org/browse/languages/ang", max_depth=3)
    assert len(results) == 1
    assert results[0].document_url == "https://www.gutenberg.org/ebooks/111.txt.utf-8"
    assert results[0].category == "bilingual_reference"
    assert results[0].found_via_chain == [
        "https://www.gutenberg.org/browse/languages/ang",
        "https://www.gutenberg.org/ebooks/111",
    ]


def test_crawl_downgrades_follow_to_uncertain_at_max_depth(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        survey,
        "fetch_source",
        lambda url: FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url),
    )
    monkeypatch.setattr(
        survey, "classify_links", lambda links, **k: [survey.LinkDecision(decision="follow", reasoning="maybe")]
    )
    results = survey.crawl("https://example.com/index", max_depth=0)
    assert len(results) == 1
    assert results[0].category == "uncertain"
    assert "max_depth" in results[0].reasoning


def test_crawl_never_follows_a_gutenberg_author_listing_even_when_decided_follow(monkeypatch):
    # Confirmed live: this was the actual cost driver of a real run (1,418 calls / ~$25.62) --
    # a correct "follow" decision on an author-name link cascading into their entire
    # bibliography. The override must fire regardless of what classify_links decided.
    _no_sleep(monkeypatch)
    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        return FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url)

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(
        survey,
        "classify_links",
        lambda links, **k: [survey.LinkDecision(decision="follow", reasoning="classical scholar, likely more works")],
    )
    author_link = PageLink(url="https://www.gutenberg.org/browse/authors/u#a49", text="Some Author", context="ctx")
    monkeypatch.setattr(survey, "extract_links", lambda html, url: [author_link])

    results = survey.crawl("https://example.com/index", max_depth=5)
    assert len(fetch_calls) == 1  # the author-listing URL was never fetched
    assert len(results) == 1
    assert results[0].category == "uncertain"
    assert "author bibliography" in results[0].reasoning


def test_visit_page_skips_classification_for_a_broad_listing(monkeypatch):
    # Confirmed by the cost arithmetic in the plan: a page with more links than one batch
    # gets ONE cheap page-scope call instead of 2+ full classify_links batches when that
    # scope check says "broad_listing" -- classify_links must never be called for its links.
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        survey,
        "fetch_source",
        lambda url: FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url),
    )
    many_links = [PageLink(url=f"https://example.com/{i}", text=f"Work {i}", context="ctx") for i in range(20)]
    monkeypatch.setattr(survey, "extract_links", lambda html, url: many_links)
    monkeypatch.setattr(
        survey,
        "_page_scope",
        lambda html, url, **k: survey.PageScopeDecision(decision="broad_listing", reasoning="wide"),
    )
    classify_calls = []
    monkeypatch.setattr(survey, "classify_links", lambda links, **k: classify_calls.append(links) or [])

    visit = survey._visit_page("https://example.com/index", model=None)
    assert visit["kind"] == "broad_listing"
    assert visit["link_count"] == 20
    assert classify_calls == []  # never reached -- that's the whole point of the gate


def test_visit_page_proceeds_normally_for_a_single_work_with_many_links(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        survey,
        "fetch_source",
        lambda url: FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url),
    )
    many_links = [PageLink(url=f"https://example.com/{i}", text=f"Work {i}", context="ctx") for i in range(20)]
    monkeypatch.setattr(survey, "extract_links", lambda html, url: many_links)
    monkeypatch.setattr(
        survey,
        "_page_scope",
        lambda html, url, **k: survey.PageScopeDecision(decision="single_work", reasoning="narrow"),
    )
    classify_calls = []

    def fake_classify(links, **k):
        classify_calls.append(links)
        return [survey.LinkDecision(decision="irrelevant", reasoning="r") for _ in links]

    monkeypatch.setattr(survey, "classify_links", fake_classify)

    visit = survey._visit_page("https://example.com/index", model=None)
    assert visit["kind"] == "links"
    assert len(classify_calls) == 1  # the gate let it through to the normal path
    assert len(classify_calls[0]) == 20


def test_crawl_never_gates_its_own_starting_url_as_a_broad_listing(monkeypatch):
    # Confirmed live: a real run against enm's own browse page got ITS OWN 23 links judged
    # "broad listing" and never expanded at all, silently finding nothing -- the gate must
    # never apply to the URL the user explicitly asked to survey, only to pages discovered
    # mid-crawl (depth > 0).
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        survey,
        "fetch_source",
        lambda url: FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url),
    )
    many_links = [PageLink(url=f"https://example.com/{i}", text=f"Work {i}", context="ctx") for i in range(20)]
    monkeypatch.setattr(survey, "extract_links", lambda html, url: many_links)
    page_scope_calls = []
    monkeypatch.setattr(survey, "_page_scope", lambda html, url, **k: page_scope_calls.append(url) or None)
    monkeypatch.setattr(
        survey,
        "classify_links",
        lambda links, **k: [survey.LinkDecision(decision="irrelevant", reasoning="r") for _ in links],
    )

    survey.crawl("https://example.com/index", max_depth=1)
    assert page_scope_calls == []  # never called for depth 0, the starting URL itself


def test_page_scope_builds_prompt_from_title_and_excerpt_only(monkeypatch):
    html = "<html><head><title>A Specific Work</title></head><body><p>Some body text.</p></body></html>"
    prompts = []
    monkeypatch.setattr(
        survey,
        "call_llm_json",
        lambda prompt, **k: prompts.append(prompt) or {"decision": "single_work", "reasoning": "ok"},
    )
    decision = survey._page_scope(html, "https://example.com/work", model=None)
    assert decision.decision == "single_work"
    assert "A Specific Work" in prompts[0]
    assert "Some body text." in prompts[0]


def test_crawl_respects_max_pages_budget(monkeypatch):
    _no_sleep(monkeypatch)
    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        return FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url)

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(
        survey,
        "classify_links",
        lambda links, **k: [survey.LinkDecision(decision="follow", reasoning="keep going") for _ in links],
    )
    survey.crawl("https://example.com/index", max_depth=10, max_pages=2)
    assert len(fetch_calls) == 2


def test_crawl_stops_gracefully_at_max_cost_usd(monkeypatch):
    # Confirmed live: max_pages/max_depth alone didn't bound $ spent when "follow" cascaded
    # through author-bibliography pages -- max_cost_usd is the safety valve that actually
    # matches what went wrong. It must stop WITHOUT raising, keeping whatever was found.
    _no_sleep(monkeypatch)
    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        return FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url)

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(
        survey,
        "classify_links",
        lambda links, **k: [survey.LinkDecision(decision="follow", reasoning="keep going") for _ in links],
    )
    # Simulated spend: $0 before the crawl starts and before the first page, then $2 from
    # the second loop check onward -- so exactly one page gets visited before the $1 budget
    # check (at the top of the *next* iteration) stops the crawl.
    usage_calls = {"n": 0}
    simulated_costs = [0.0, 0.0, 2.0]

    def fake_usage():
        idx = min(usage_calls["n"], len(simulated_costs) - 1)
        usage_calls["n"] += 1
        return {"cost_usd": simulated_costs[idx]}

    monkeypatch.setattr(survey, "get_usage_summary", fake_usage)

    results = survey.crawl("https://example.com/index", max_depth=10, max_pages=300, max_cost_usd=1.0)
    assert len(fetch_calls) == 1  # stopped after one page, not hundreds
    assert results == []  # that one page's "follow" link was still queued, never raised


def test_crawl_never_revisits_the_same_url_twice(monkeypatch):
    _no_sleep(monkeypatch)
    html_with_duplicate_link = """
    <a href="/same">dup one</a>
    <a href="/same">dup two (different anchor text, same URL)</a>
    """
    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        if url == "https://example.com/index":
            return FetchedContent(raw=html_with_duplicate_link.encode(), content_type="text/html", final_url=url)
        return FetchedContent(raw=b"content", content_type="text/plain", final_url=url)

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(
        survey, "classify_links", lambda links, **k: [survey.LinkDecision(decision="candidate", reasoning="dup")]
    )
    results = survey.crawl("https://example.com/index", max_depth=2)
    assert fetch_calls.count("https://example.com/same") <= 1
    assert len(results) == 1


def test_crawl_is_resumable_via_checkpoint(monkeypatch, tmp_path):
    _no_sleep(monkeypatch)
    fetch_calls = []

    def fake_fetch(url):
        fetch_calls.append(url)
        return FetchedContent(raw=b"content", content_type="text/plain", final_url=url)

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    checkpoint = Checkpoint(tmp_path / "progress.json", meta={})
    first = survey.crawl("https://example.com/book.txt", checkpoint=checkpoint)
    second = survey.crawl("https://example.com/book.txt", checkpoint=checkpoint)
    assert first == second
    assert len(fetch_calls) == 1  # second crawl served entirely from the checkpoint


def test_crawl_survives_one_pages_malformed_reply_without_losing_others(monkeypatch):
    # Confirmed live: a single batch's malformed reply used to raise out of crawl()
    # entirely, discarding every result already found elsewhere in the frontier --
    # real money, zero usable output, on an actual run against all six real pages.
    _no_sleep(monkeypatch)

    def fake_fetch(url):
        if url == "https://example.com/index":
            html = '<a href="/ok">An ok page</a><a href="/broken">A broken page</a>'
            return FetchedContent(raw=html.encode(), content_type="text/html", final_url=url)
        if url == "https://example.com/ok":
            html = '<a href="/ok-child">a fine old manuscript</a>'
            return FetchedContent(raw=html.encode(), content_type="text/html", final_url=url)
        if url == "https://example.com/broken":
            html = '<a href="/trigger-failure">TRIGGER_FAILURE</a>'
            return FetchedContent(raw=html.encode(), content_type="text/html", final_url=url)
        if url == "https://example.com/ok-child":
            return FetchedContent(raw=b"content", content_type="text/plain", final_url=url)
        raise AssertionError(f"unexpected fetch: {url}")

    def fake_classify(links, **k):
        if any("TRIGGER_FAILURE" in link.text for link in links):
            raise ValueError("expected a JSON array of 1 entries, got []")  # simulates the real live failure
        return [survey.LinkDecision(decision="follow", reasoning="fine") for _ in links]

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(survey, "classify_links", fake_classify)

    results = survey.crawl("https://example.com/index", max_depth=3)

    # The /ok branch must still have been fully explored and reported...
    assert any(r.document_url == "https://example.com/ok-child" for r in results)
    # ...and the /broken branch must be flagged for manual review, not silently dropped
    # or allowed to abort the whole crawl.
    assert any("classification failed" in r.reasoning for r in results)


def test_crawl_survives_an_llm_error_not_just_valueerror(monkeypatch):
    # Confirmed live, a second time, after the ValueError case above was already fixed: a
    # real run against `la` made 96 successful calls, then hit one `LLMEmptyResponseError`
    # (a RuntimeError subclass, NOT a ValueError -- the claude CLI backend returned an empty/
    # truncated response for one call) -- which this crawl's `except ValueError` didn't catch,
    # discarding all 96 calls' worth of results the same way the first bug did.
    _no_sleep(monkeypatch)

    def fake_fetch(url):
        if url == "https://example.com/index":
            html = '<a href="/ok">An ok page</a><a href="/broken">A broken page</a>'
            return FetchedContent(raw=html.encode(), content_type="text/html", final_url=url)
        if url == "https://example.com/ok":
            return FetchedContent(raw=b"content", content_type="text/plain", final_url=url)
        if url == "https://example.com/broken":
            html = '<a href="/trigger-failure">TRIGGER_FAILURE</a>'
            return FetchedContent(raw=html.encode(), content_type="text/html", final_url=url)
        raise AssertionError(f"unexpected fetch: {url}")

    def fake_classify(links, **k):
        if any("TRIGGER_FAILURE" in link.text for link in links):
            raise LLMEmptyResponseError("model response was empty or looked truncated")
        return [survey.LinkDecision(decision="follow", reasoning="fine") for _ in links]

    monkeypatch.setattr(survey, "fetch_source", fake_fetch)
    monkeypatch.setattr(survey, "classify_links", fake_classify)

    results = survey.crawl("https://example.com/index", max_depth=3)

    assert any(r.document_url == "https://example.com/ok" for r in results)
    assert any("classification failed" in r.reasoning for r in results)


def test_crawl_drops_irrelevant_links_without_emitting_a_result(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(
        survey,
        "fetch_source",
        lambda url: FetchedContent(raw=_NON_GUTENBERG_HTML.encode(), content_type="text/html", final_url=url),
    )
    monkeypatch.setattr(
        survey, "classify_links", lambda links, **k: [survey.LinkDecision(decision="irrelevant", reasoning="nope")]
    )
    results = survey.crawl("https://example.com/index", max_depth=2)
    assert results == []
