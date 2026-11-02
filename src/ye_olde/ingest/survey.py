"""Generalized, recursive index-page crawler for finding old-language
primary-text candidates on any digitization site (Project Gutenberg's
language-browse pages are the first real case, but nothing here is
Gutenberg-specific except two small optional shortcuts -- see
`_gutenberg_format_links`/`_gutenberg_bibrec_hint`).

The pipeline so far only runs forward from an already-known document URL
(`acquire.py`). This module runs the other direction: starting from any
index-of-works page, it follows promising links hop by hop -- the LLM
decides which links are worth following deeper, which are already
document-shaped candidates, and which are irrelevant -- until it reaches
an actual document URL, or a configured depth/page budget is spent.

Every failure here (a bad HTTP status, a malformed LLM reply) raises
rather than being caught -- domain code, not a boundary;
`scripts/survey_catalog.py`'s `main()` is the boundary (CLAUDE.md "Error
handling").

**What this deliberately does not do (v1 scope, stated outright rather
than discovered later -- same posture as `acquire.py`'s one-URL-per-edition
assumption and `split_mixed.py`'s one-run-per-bootstrap limitation):**
- Does not follow pagination within a single listing automatically --
  each starting page is crawled once; a paginated index needs each page
  passed in as its own starting URL.
- Does not detect a pure "read online" HTML page (no separate raw-text
  download, the HTML page itself *is* the document -- the case
  `acquire.py`'s own HTML path already handles once a URL is known) as a
  terminal document automatically. An HTML page with no links worth
  following is flagged `uncertain` instead of guessing it's the document
  itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

from ..common.errors import LLMError
from ..common.llm_client import call_llm_json, get_usage_summary
from ..common.logging import get_logger
from ..prompt import load_prompt
from .acquire import classify_content, fetch_source
from .checkpoint import Checkpoint

_log = get_logger(__name__)

_SYSTEM_PROMPT = load_prompt("ingest", "survey_classify_system")
_PAGE_SCOPE_SYSTEM_PROMPT = load_prompt("ingest", "survey_page_scope_system")

_DEFAULT_MAX_DEPTH = 3
_DEFAULT_MAX_PAGES = 300
_DEFAULT_BATCH_SIZE = 15
# Confirmed live: three modest-looking starting pages (ang/enm/grc) alone ran up 1,418 LLM
# calls (~$25.62) via "follow" cascading through author-bibliography pages, each listing
# many more works that themselves got "follow"ed -- page/depth limits bound breadth, but
# say nothing about $ spent. A hard dollar ceiling is the safety valve that actually matches
# what went wrong: it stops the crawl (gracefully, keeping whatever was already found) the
# moment combinatorial fan-out gets expensive, regardless of how many pages that took.
_DEFAULT_MAX_COST_USD = 3.0

# A courtesy delay before every real fetch (never on a checkpoint cache hit -- `_visit_page`
# is only ever called for an actual network request, see `crawl`) -- a multi-hundred-page
# crawl hitting one site back-to-back as fast as the network allows is a bad way to treat
# someone else's server, independent of this project's own retry budget in `acquire.py`.
_POLITE_DELAY_SECONDS = 0.5

Decision = Literal["follow", "candidate", "irrelevant", "uncertain"]
Category = Literal["old_text_candidate", "bilingual_reference"]


@dataclass(frozen=True)
class PageLink:
    url: str
    text: str
    context: str


def extract_links(html: str, base_url: str) -> list[PageLink]:
    """Every `<a href>` on the page that isn't site chrome, resolved to an
    absolute URL, paired with its own link text and its parent block's text
    (a hint might sit next to a link rather than inside it -- e.g.
    Gutenberg's own `(language) (as role)` parentheticals). Deliberately
    DOM-based (`find_parent`), not a site-specific CSS class -- this has to
    make sense of whatever markup an arbitrary site uses, not just
    Gutenberg's.

    Two things are dropped before a link ever reaches the LLM, not after --
    this isn't a style choice, both were confirmed as real failure modes on
    an actual live run against real Gutenberg pages:
    - `nav`/`header`/`footer` content (same tags `acquire.extract_html_text`
      already strips for the same reason) -- confirmed live that Gutenberg's
      shared site navigation (donate/help/about links, present on every
      single page of the site) was consuming entire classification batches
      before the real per-page content was ever reached.
    - A link with no usable text *and* no usable context (e.g. a bare
      "skip to content" anchor or a logo-only link) -- confirmed live that
      sending one of these to the LLM is what caused it to reply with one
      fewer decision than links given (nothing to classify, nothing
      returned), silently shifting every following decision's alignment by
      one position. Not worth an LLM call either way: there's nothing in
      a contextless link for a classification decision to be based on.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    links: list[PageLink] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue
        text = a.get_text(" ", strip=True)
        parent = a.find_parent(["li", "tr", "p"])
        context = parent.get_text(" ", strip=True) if parent else text
        if not text and not context:
            continue
        seen.add(absolute)
        links.append(PageLink(url=absolute, text=text, context=context))
    return links


def looks_like_a_document(content_type: str, url: str) -> bool:
    """True when an *already-fetched* response is itself a raw downloadable
    document (`.txt`/`.pdf`) rather than an HTML page to extract links from
    -- reuses `acquire.classify_content`'s header-then-suffix logic,
    treating every non-"html" result as terminal (see module docstring for
    the one case -- a pure "read online" HTML page -- this doesn't
    auto-detect). Requires a real `content_type` from a real fetch:
    `classify_content` falls back to assuming "txt" for an unrecognized
    *and unfetched* URL, so calling this with a blank `content_type` before
    actually fetching the URL would wrongly call almost anything a
    document -- see `_url_suffix_is_a_document` for the *pre-fetch* check
    this module actually uses instead.
    """
    return classify_content(content_type, url) != "html"


def _url_suffix_is_a_document(url: str) -> bool:
    """A conservative, pre-fetch check: true only when `url` itself ends in
    `.txt`/`.pdf` -- unlike `classify_content`'s lenient "assume txt"
    fallback (the right call *after* a real fetch confirms nothing more
    specific, wrong here since nothing has been fetched yet), a link whose
    URL doesn't obviously look like a raw file is sent back through the
    frontier to actually be resolved (`_visit_page`), never guessed at.
    """
    return Path(urlparse(url).path).suffix.lower() in (".txt", ".pdf")


def _is_gutenberg_author_listing(url: str) -> bool:
    """Optional, Gutenberg-specific shortcut (like `_gutenberg_format_links`):
    true for an author's full-bibliography page (`/browse/authors/<letter>#a<id>`,
    the real shape confirmed live). `crawl` never auto-follows one of these --
    confirmed live that a `follow` decision on an author-name link cascading
    into their entire bibliography (sometimes dozens more works, each then
    needing its own classification) was the actual cost driver of a real run
    (1,418 calls / ~$25.62 for three modest starting pages), not misclassified
    junk. `False` on every other URL, including every non-Gutenberg site, so
    nothing elsewhere depends on this ever firing.
    """
    return urlparse(url).path.startswith("/browse/authors/")


def _gutenberg_format_links(html: str) -> list[str]:
    """Optional Gutenberg-specific shortcut: an `/ebooks/<id>` bibrec page
    lists its direct download formats as `<a class="featured-format-link"|
    "other-format-link" type="...">` (verified against a real fetched
    page). Returns the `.txt`/`.pdf` ones, absolute, if any -- `[]` on any
    page without that structure (every non-Gutenberg site, and any
    Gutenberg page that isn't a bibrec page), so the rest of the crawl
    falls through to the generic LLM-guided path exactly as if this
    function didn't exist.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", class_=["featured-format-link", "other-format-link"], href=True):
        mime = (a.get("type") or "").split(";")[0].strip().lower()
        if mime in ("text/plain", "application/pdf"):
            links.append(urljoin("https://www.gutenberg.org", a["href"]))
    return links


def _gutenberg_bibrec_hint(html: str) -> str | None:
    """Optional Gutenberg-specific bonus context: renders a `/ebooks/<id>`
    page's `#bibrec` table (author names -- which already include
    birth/death years in Gutenberg's own display text, e.g. "Aelfric,
    Abbot of Eynsham, 955?-1020?" -- roles, every language tag, and any
    LCSH subject line) as one paragraph, to sharpen the next classification
    call. `None` on any page without a `#bibrec` table.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(id="bibrec")
    if not isinstance(table, Tag):
        return None
    parts: list[str] = []
    for row in table.find_all("tr"):
        header = row.find("th")
        cell = row.find("td")
        if header is None or cell is None:
            continue
        label = header.get_text(" ", strip=True)
        if label in ("Author", "Editor", "Translator", "Title", "Language", "Subject"):
            parts.append(f"{label}: {cell.get_text(' ', strip=True)}")
    return "; ".join(parts) if parts else None


class LinkDecision(BaseModel):
    decision: Decision
    category: Category | None = None
    related_to: str = ""
    reasoning: str = ""
    confidence: float | None = None


def classify_links(
    links: list[PageLink], *, model: str | None = None, batch_size: int = _DEFAULT_BATCH_SIZE
) -> list[LinkDecision]:
    """One `LinkDecision` per `links` entry, via `survey_classify_system`
    (`prompt/ingest.yaml`), batched `call_llm_json` calls. A batch whose
    reply isn't a same-length JSON array raises -- misclassifying which
    links are worth following is a correctness bug, not a stylistic nit
    (same reasoning as `split_mixed.classify_chunks`).

    A batch of exactly one link is normalized up-front if the reply came
    back as a bare JSON object instead of a one-element array -- confirmed
    live that a real model, asked for "an array the same length as the
    input" with an input of length 1, sometimes "helpfully" unwraps it to a
    single object anyway. This matters here specifically (not just a
    one-off): `_visit_page`'s Gutenberg-format-link path always classifies
    exactly one pseudo-link, so this is that path's *common* case, not a
    rare edge case worth only a fallback.

    A count mismatch at any batch size gets exactly one repair retry --
    confirmed live (a real run against the six actual language pages) that
    this still happens at larger batch sizes too, not just the one-link
    case above: 22 of 1418 calls in that run undercounted by one, most
    plausibly the model treating two distinct-but-similarly-worded links
    (e.g. two different 19th-century editions of the same classical work)
    as one. The retry shows the model its own reply and asks for the exact
    count back, the same "exactly one repair attempt" shape
    `common.llm_client.call_llm_json` already uses for a malformed-JSON
    reply -- this is the analogous repair for a malformed-*shape* reply,
    which that function's own JSON-syntax repair doesn't cover. A second
    failure still raises -- a repair call is not obviously more likely to
    succeed a second time, so retrying indefinitely isn't worth it.
    """
    results: list[LinkDecision] = []
    for start in range(0, len(links), batch_size):
        batch = links[start : start + batch_size]
        prompt = "\n".join(f"{i}. TEXT: {link.text}\n   CONTEXT: {link.context}" for i, link in enumerate(batch))
        reply = _classify_batch_with_repair(prompt, batch_size=len(batch), model=model)
        results.extend(LinkDecision.model_validate(item) for item in reply)
    return results


def _normalize_batch_reply(reply: object, *, batch_size: int) -> list[object] | None:
    if batch_size == 1 and isinstance(reply, dict):
        reply = [reply]
    if isinstance(reply, list) and len(reply) == batch_size:
        return reply
    return None


def _classify_batch_with_repair(prompt: str, *, batch_size: int, model: str | None) -> list[object]:
    """Classifies one batch, with one repair retry on a count mismatch (see
    `classify_links`'s docstring). If the repair retry *also* comes back the
    wrong shape, this batch degrades to `batch_size` "uncertain" placeholders
    instead of raising -- confirmed live, at real scale (a 146-link page
    needing ~10 batches), that raising here is actively harmful: `classify_links`
    only returns its accumulated `results` once every batch succeeds, so one
    unrecoverable batch anywhere in that loop silently discarded every other
    batch's already-correct classifications too, not just its own. A real run
    lost ~145 links' worth of real work this way over exactly one bad batch.
    Degrading just the broken batch is strictly better: the other 9 batches'
    results survive, and the few links in this one batch are flagged for a
    human instead of guessed at.
    """
    reply = call_llm_json(prompt, system=_SYSTEM_PROMPT, model=model)
    normalized = _normalize_batch_reply(reply, batch_size=batch_size)
    if normalized is not None:
        return normalized

    repair_prompt = (
        f"{prompt}\n\n---\n\nYour previous reply was: {reply!r}\n\nThat has the wrong number of entries. "
        f"I gave you exactly {batch_size} numbered links above and need exactly {batch_size} decisions back, "
        "same order -- one per link, even if two links look similar or redundant to you; never merge or "
        "drop one. Reply again with ONLY the corrected JSON array, no prose, no markdown code fence."
    )
    reply = call_llm_json(repair_prompt, system=_SYSTEM_PROMPT, model=model)
    normalized = _normalize_batch_reply(reply, batch_size=batch_size)
    if normalized is not None:
        return normalized

    _log.warning("batch of %d links still malformed after repair retry, degrading to uncertain: %r", batch_size, reply)
    return [
        {"decision": "uncertain", "reasoning": f"classification failed for this link even after retry: {reply!r}"}
        for _ in range(batch_size)
    ]


class PageScopeDecision(BaseModel):
    decision: Literal["single_work", "broad_listing"]
    reasoning: str = ""


def _page_scope(html: str, url: str, *, model: str | None) -> PageScopeDecision:
    """One cheap `call_llm_json` call -- the page's own `<title>` plus a short
    visible-text excerpt, no link list -- deciding whether a page is a single
    work's own page or a broad listing not worth auto-expanding. Only called
    when `extract_links` found more links than fit in one `classify_links`
    batch (see its call site in `_visit_page`): within one batch, classifying
    every link directly costs the same as this check would, so there's
    nothing to save by adding it there. Site-agnostic (title/excerpt only,
    no Gutenberg-specific markup) -- the general counterpart to
    `_is_gutenberg_author_listing`'s free but Gutenberg-only shortcut.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    excerpt = soup.get_text(" ", strip=True)[:500]
    prompt = f"TITLE: {title}\nEXCERPT: {excerpt}"
    reply = call_llm_json(prompt, system=_PAGE_SCOPE_SYSTEM_PROMPT, model=model)
    return PageScopeDecision.model_validate(reply)


class SurveyResult(BaseModel):
    """One actionable outcome of the crawl -- crosses into the human-facing
    output files, so a `BaseModel` (CLAUDE.md "Pydantic"), same reasoning
    as `acquire.SourceEntry`.
    """

    document_url: str | None
    found_via_chain: list[str]
    title: str
    category: Literal["old_text_candidate", "bilingual_reference", "uncertain"]
    reasoning: str
    related_to: str = ""


class _LinkOutcome(BaseModel):
    url: str
    text: str
    decision: Decision
    category: Category | None = None
    reasoning: str = ""
    related_to: str = ""


class _PageVisit(BaseModel):
    """The validated shape of one `_visit_page` outcome -- `crawl` reads
    this (never a raw dict) regardless of whether it came from a fresh
    `compute()` call or a `Checkpoint` reload (plain JSON-parsed data
    either way; `model_validate` is what turns either one into something
    with real typed attribute access, rather than every call site indexing
    an untyped `dict[str, object]` and needing its own `# type: ignore`).
    """

    kind: Literal["document", "resolved", "no_links", "broad_listing", "links"]
    url: str | None = None
    title: str | None = None
    decision: Decision | None = None
    category: Category | None = None
    reasoning: str = ""
    related_to: str = ""
    link_count: int = 0
    links: list[_LinkOutcome] = []


def _visit_page(url: str, *, model: str | None, allow_broad_listing_gate: bool = True) -> dict[str, object]:
    """Fetches and judges exactly one page -- the unit of work cached per
    URL by `crawl`'s checkpoint, so re-running a long crawl never re-fetches
    or re-classifies a page it already has an answer for.

    `allow_broad_listing_gate=False` for the crawl's own starting URL (see
    `crawl`'s call site) -- a wide listing is exactly what a starting URL is
    *expected* to be (that's the whole reason to start a crawl there), so
    the "don't auto-expand a wide listing" judgment only makes sense for a
    page *discovered* mid-crawl, never the one the user explicitly asked to
    survey. Confirmed live this was a real bug, not a hypothetical one: a
    fresh run against `enm`'s own browse page got its own 23 links judged
    "broad listing" and never expanded at all, silently finding nothing.

    Returns a plain JSON-serializable dict (`Checkpoint` persists via
    `json.dumps`, so this can't return a `_PageVisit`/`LinkDecision`
    instance directly) -- `crawl` immediately re-validates it into a
    `_PageVisit` before reading anything out of it.
    """
    fetched = fetch_source(url)
    time.sleep(_POLITE_DELAY_SECONDS)
    if looks_like_a_document(fetched.content_type, url):
        return _PageVisit(kind="document").model_dump()

    html = fetched.raw.decode("utf-8", errors="replace")
    format_links = _gutenberg_format_links(html)
    if format_links:
        title_tag = BeautifulSoup(html, "html.parser").find("title")
        title = title_tag.get_text(strip=True) if title_tag else url
        hint = _gutenberg_bibrec_hint(html) or ""
        pseudo_link = PageLink(url=format_links[0], text=title, context=hint)
        decision = classify_links([pseudo_link], model=model)[0]
        return _PageVisit(
            kind="resolved",
            url=pseudo_link.url,
            title=pseudo_link.text,
            decision=decision.decision,
            category=decision.category,
            reasoning=decision.reasoning,
            related_to=decision.related_to,
        ).model_dump()

    links = extract_links(html, url)
    if not links:
        return _PageVisit(kind="no_links").model_dump()

    if allow_broad_listing_gate and len(links) > _DEFAULT_BATCH_SIZE:
        # Worth a cheap page-level check before paying for 2+ full classify_links batches --
        # within one batch, classifying every link directly costs the same as this check would,
        # so it's only gated here, not on every page (see _page_scope's own docstring).
        scope = _page_scope(html, url, model=model)
        if scope.decision == "broad_listing":
            return _PageVisit(kind="broad_listing", reasoning=scope.reasoning, link_count=len(links)).model_dump()

    bib_hint = _gutenberg_bibrec_hint(html)
    if bib_hint:
        links = [PageLink(url=link.url, text=link.text, context=f"{link.context} | {bib_hint}") for link in links]
    decisions = classify_links(links, model=model)
    return _PageVisit(
        kind="links",
        links=[
            _LinkOutcome(
                url=link.url,
                text=link.text,
                decision=decision.decision,
                category=decision.category,
                reasoning=decision.reasoning,
                related_to=decision.related_to,
            )
            for link, decision in zip(links, decisions, strict=True)
        ],
    ).model_dump()


@dataclass(frozen=True)
class _FrontierItem:
    url: str
    depth: int
    chain: tuple[str, ...]
    # Carried over from a "candidate" decision one hop up, when that candidate's own URL
    # wasn't already obviously a document and needed one more hop to resolve -- so that
    # evidence isn't discarded just because the literal file link took one extra fetch to
    # find. `None` for a plain "follow" with no category opinion yet.
    pending_category: Category | None = None
    pending_reasoning: str = ""
    pending_related_to: str = ""


def crawl(
    start_url: str,
    *,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_pages: int = _DEFAULT_MAX_PAGES,
    max_cost_usd: float = _DEFAULT_MAX_COST_USD,
    model: str | None = None,
    checkpoint: Checkpoint | None = None,
) -> list[SurveyResult]:
    """Recursively follows `start_url`'s links, hop by hop, until each
    branch resolves to a document, is judged irrelevant, or hits
    `max_depth`/`max_pages`/`max_cost_usd` for this one starting URL.

    `chain` (the sequence of URLs that led to each result) is tracked for
    traceability but the visited-set itself is keyed on URL alone, globally
    across the whole crawl -- a URL reachable by two different paths is
    still only ever fetched/classified once.

    `max_cost_usd` is checked against `get_usage_summary()`'s running total
    *delta since this call started* -- that total is process-wide (shared
    with whatever else in this process calls an LLM), so a snapshot at
    entry is what makes the check mean "this crawl," not "this process."
    Hitting it stops gracefully (returns what's already been found,
    doesn't raise) -- the budget is a safety valve, not an error.
    """
    frontier: list[_FrontierItem] = [_FrontierItem(start_url, 0, (start_url,))]
    visited: set[str] = set()
    results: list[SurveyResult] = []
    pages_fetched = 0
    cost_at_start = float(get_usage_summary()["cost_usd"])

    while frontier and pages_fetched < max_pages:
        if float(get_usage_summary()["cost_usd"]) - cost_at_start >= max_cost_usd:
            _log.warning(
                "crawl for %r stopped: hit max_cost_usd=%.2f with %d page(s) left to visit",
                start_url,
                max_cost_usd,
                len(frontier),
            )
            break
        item = frontier.pop(0)
        url, depth, chain = item.url, item.depth, item.chain
        if url in visited:
            continue
        visited.add(url)
        pages_fetched += 1

        def compute(url: str = url, depth: int = depth) -> dict[str, object]:
            return _visit_page(url, model=model, allow_broad_listing_gate=depth > 0)

        try:
            raw_visit = checkpoint.get_or_compute(f"page:{url}", compute) if checkpoint else compute()
        except (ValueError, LLMError) as exc:
            # A malformed/empty LLM reply for this one page (CLAUDE.md "Error handling": a
            # specific, expected condition, same shape as clean.review_units_with_llm's own
            # fallback) -- confirmed live, twice, that losing an entire multi-hundred-page
            # crawl's already-found results over one bad call is a real, expensive failure
            # mode, not a hypothetical one: first a `ValueError` from a persistent batch-count
            # mismatch, then (after that was fixed) an `LLMEmptyResponseError` -- a `RuntimeError`
            # subclass, not a `ValueError`, from the `claude` CLI backend returning an empty/
            # truncated response for one call out of 96 -- which this `except` didn't catch
            # until now, and which discarded that entire crawl's results the same way. `LLMError`
            # is this project's own catch-all for "the configured LLM didn't give us what we
            # needed" (see `common/errors.py`) -- the right-sized net for this, not bare
            # `Exception`, which would also swallow a genuine bug here instead of surfacing it.
            # Logged and flagged for a human instead of silently retried (a repeat call is not
            # obviously more likely to succeed) or allowed to discard everything else.
            _log.warning("page visit failed for %r, flagging for manual review instead of aborting: %s", url, exc)
            results.append(
                SurveyResult(
                    document_url=None,
                    found_via_chain=list(chain),
                    title=url,
                    category="uncertain",
                    reasoning=f"classification failed for this page: {exc}",
                )
            )
            continue
        visit = _PageVisit.model_validate(raw_visit)

        if visit.kind == "document":
            results.append(
                SurveyResult(
                    document_url=url,
                    found_via_chain=list(chain),
                    title=url,
                    category=item.pending_category or "uncertain",
                    reasoning=item.pending_reasoning,
                    related_to=item.pending_related_to,
                )
            )
        elif visit.kind == "resolved":
            # visit.url/.title/.decision are always set for "resolved" (see _visit_page) --
            # asserted rather than silently handled, since a "resolved" visit missing them
            # would mean _visit_page itself is broken, not a normal input-shape variation.
            assert visit.url is not None and visit.title is not None and visit.decision is not None
            if visit.decision in ("candidate", "follow", "uncertain"):
                results.append(
                    SurveyResult(
                        document_url=visit.url,
                        found_via_chain=list(chain),
                        title=visit.title,
                        category=visit.category or "uncertain",
                        reasoning=visit.reasoning,
                        related_to=visit.related_to,
                    )
                )
            # "irrelevant" -> nothing emitted; already recorded in the checkpoint via `visit`.
        elif visit.kind == "no_links":
            results.append(
                SurveyResult(
                    document_url=None,
                    found_via_chain=list(chain),
                    title=url,
                    category="uncertain",
                    reasoning="HTML page with no links to classify -- may itself be a 'read online' document.",
                )
            )
        elif visit.kind == "broad_listing":
            results.append(
                SurveyResult(
                    document_url=None,
                    found_via_chain=list(chain),
                    title=url,
                    category="uncertain",
                    reasoning=(
                        f"broad listing ({visit.link_count} links), not auto-expanded -- survey it "
                        f"directly as its own starting URL if its full contents are wanted. "
                        f"{visit.reasoning}"
                    ),
                )
            )
        elif visit.kind == "links":
            for link in visit.links:
                if link.decision == "follow":
                    if _is_gutenberg_author_listing(link.url):
                        results.append(
                            SurveyResult(
                                document_url=None,
                                found_via_chain=[*chain, link.url],
                                title=link.text,
                                category="uncertain",
                                reasoning=(
                                    "author bibliography page, not auto-followed (can list dozens of "
                                    f"unrelated works) -- survey it directly as its own starting URL if "
                                    f"its full bibliography is wanted. Original reasoning: {link.reasoning}"
                                ),
                            )
                        )
                    elif depth < max_depth:
                        frontier.append(_FrontierItem(link.url, depth + 1, (*chain, link.url)))
                    else:
                        results.append(
                            SurveyResult(
                                document_url=None,
                                found_via_chain=[*chain, link.url],
                                title=link.text,
                                category="uncertain",
                                reasoning=f"max_depth reached while still 'follow'-worthy: {link.reasoning}",
                            )
                        )
                elif link.decision == "candidate":
                    if _url_suffix_is_a_document(link.url):
                        results.append(
                            SurveyResult(
                                document_url=link.url,
                                found_via_chain=[*chain, link.url],
                                title=link.text,
                                category=link.category or "uncertain",
                                reasoning=link.reasoning,
                                related_to=link.related_to,
                            )
                        )
                    elif depth < max_depth:
                        # Not obviously a raw file yet -- one more hop to actually resolve it
                        # (e.g. a Gutenberg ebook page still needs its own bibrec visited to
                        # find the real .txt/.pdf link), carrying this category/reasoning
                        # forward rather than discarding it once that hop confirms a document.
                        frontier.append(
                            _FrontierItem(
                                link.url,
                                depth + 1,
                                (*chain, link.url),
                                pending_category=link.category,
                                pending_reasoning=link.reasoning,
                                pending_related_to=link.related_to,
                            )
                        )
                    else:
                        results.append(
                            SurveyResult(
                                document_url=None,
                                found_via_chain=[*chain, link.url],
                                title=link.text,
                                category=link.category or "uncertain",
                                reasoning=link.reasoning,
                                related_to=link.related_to,
                            )
                        )
                elif link.decision == "uncertain":
                    results.append(
                        SurveyResult(
                            document_url=None,
                            found_via_chain=[*chain, link.url],
                            title=link.text,
                            category="uncertain",
                            reasoning=link.reasoning,
                        )
                    )
                # "irrelevant" -> nothing emitted.
    return results
