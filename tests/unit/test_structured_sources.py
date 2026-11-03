"""Unit tests for structured_sources.py: Perseus CTS parsing, TEI header
extraction, free-text fallback parsing, AI-draft provenance, the
Wikidata/LLM composition-year lookup, and the end-to-end resolver -- all
against mocked HTTP/LLM calls, no live network.
"""

from __future__ import annotations

import httpx
import pytest

from ye_olde.ingest import structured_sources as ss

_CTS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ti:work xmlns:ti="http://chs.harvard.edu/xmlns/cts" urn="urn:cts:latinLit:phi0690.phi003" xml:lang="lat">
    <ti:title xml:lang="eng">Aeneid</ti:title>
    <ti:edition urn="urn:cts:latinLit:phi0690.phi003.perseus-lat2" xml:lang="lat">
        <ti:label xml:lang="eng">Latin edition</ti:label>
        <ti:description xml:lang="eng">Vergil. The Bucolics, Aeneid, and Georgics Of Virgil. Greenough, J.B., editor. Boston: Ginn and Company, 1881.</ti:description>
    </ti:edition>
    <ti:translation urn="urn:cts:latinLit:phi0690.phi003.perseus-eng2" xml:lang="eng">
        <ti:label xml:lang="eng">English translation</ti:label>
        <ti:description xml:lang="eng">Vergil. The Aeneid of Virgil. Williams, Theodore, C, translator. Boston: Houghton Mifflin, 1910.</ti:description>
    </ti:translation>
    <ti:commentary urn="urn:cts:latinLit:phi0690.phi003.perseus-com1" xml:lang="eng">
        <ti:label xml:lang="eng">Commentary</ti:label>
        <ti:description xml:lang="eng">Notes on the Aeneid.</ti:description>
    </ti:commentary>
</ti:work>
"""

_LAT_TEI_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Aeneid</title>
        <author>P. Vergilius Maro</author>
        <editor>J. B. Greenough</editor>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <monogr>
            <imprint><pubPlace>Boston</pubPlace><publisher>Ginn and Company</publisher><date>1881</date></imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text xml:lang="lat">
    <body>
      <div type="edition" xml:lang="lat">
        <div n="1" type="textpart">
          <l n="1">Arma virumque cano<note>a footnote</note>, Troiae qui primus ab oris</l>
          <l n="2">Italiam fato profugus</l>
        </div>
      </div>
    </body>
  </text>
</TEI>
"""

_ENG_TEI_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Aeneid</title>
        <author>P. Vergilius Maro</author>
        <editor role="translator">Theodore C. Williams</editor>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <monogr>
            <imprint><pubPlace>Boston</pubPlace><publisher>Houghton Mifflin</publisher><date>1910</date></imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text xml:lang="eng">
    <body>
      <div type="translation" xml:lang="eng">
        <div n="1" type="textpart">
          <l n="1">I sing of arms and the man</l>
        </div>
      </div>
    </body>
  </text>
</TEI>
"""


def _response(status_code: int = 200, **kwargs: object) -> httpx.Response:
    """`httpx.Response(...)` with a dummy request attached -- `raise_for_status()`
    requires one even on a 200, even though none of these tests care which URL it names.
    """
    return httpx.Response(status_code, request=httpx.Request("GET", "https://example.test"), **kwargs)  # type: ignore[arg-type]


def _fake_httpx_get(routes: dict[str, httpx.Response]):
    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        for prefix, response in routes.items():
            if url.startswith(prefix):
                return response
        raise AssertionError(f"unexpected URL in test: {url}")

    return fake_get


# --- URL matching -----------------------------------------------------------


def test_is_perseus_cts_url_matches_real_shape():
    assert ss._is_perseus_cts_url(
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/__cts__.xml"
    )


def test_is_perseus_cts_url_rejects_other_sources():
    assert not ss._is_perseus_cts_url("https://www.gutenberg.org/cache/epub/123/pg123.txt")
    assert not ss._is_perseus_cts_url(
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/__cts__.xml".replace(
            "phi0690/__cts__.xml", "phi0690/phi003/not_cts.xml"
        )
    )


# --- CTS parsing -------------------------------------------------------------


def test_parse_cts_work_extracts_title_role_lang_and_description():
    work_lang, title, entries = ss._parse_cts_work(_CTS_XML)
    assert work_lang == "lat"
    assert title == "Aeneid"
    roles = {e.role: e for e in entries}
    assert set(roles) == {"edition", "translation", "commentary"}
    assert roles["edition"].lang == "lat"
    assert roles["translation"].lang == "eng"
    assert "translator" in roles["translation"].description


def test_parse_xml_tolerates_undefined_html_entities():
    """Several older canonical-latinLit source files use bare HTML named
    entities (&dagger;, &mdash;, &lsquo;, ...) that plain XML doesn't
    define -- a real, caught-live ParseError otherwise.
    """
    xml = b"<root>Some note&dagger; and an &mdash; and &lsquo;a quote&rsquo;.</root>"
    root = ss._parse_xml(xml)
    assert root.text == "Some note† and an — and ‘a quote’."


def test_parse_xml_leaves_xml_predefined_entities_alone():
    xml = b"<root>a &amp; b &lt; c</root>"
    root = ss._parse_xml(xml)
    assert root.text == "a & b < c"


# --- TEI text/header extraction ---------------------------------------------


def test_extract_tei_text_joins_lines_and_drops_notes():
    text = ss._extract_tei_text(_LAT_TEI_XML)
    assert text == "Arma virumque cano, Troiae qui primus ab oris\nItaliam fato profugus"
    assert "footnote" not in text


def test_extract_tei_text_raises_without_body():
    with pytest.raises(ValueError):
        ss._extract_tei_text(b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/></TEI>')


_LEGACY_TEI2_XML = b"""<TEI.2>
  <teiHeader>
    <fileDesc><titleStmt><author>Claudian</author><editor role="translator">Maurice Platnauer</editor></titleStmt>
      <sourceDesc><biblStruct><monogr><imprint><date>1922</date></imprint></monogr></biblStruct></sourceDesc>
    </fileDesc>
  </teiHeader>
  <text xml:lang="eng"><body><div1 type="poem"><p>Shorter poems text.</p></div1></body></text>
</TEI.2>
"""


def test_extract_tei_text_handles_legacy_unnamespaced_tei2():
    """A few older Perseus documents (seen live: canonical-latinLit's
    Claudian translation) use the legacy, unnamespaced TEI P3/P4 DTD
    (root `<TEI.2>`) instead of the modern TEI P5 namespace -- a plain
    namespaced `.//{tei-ns}body` search finds nothing on one of these.
    """
    assert ss._extract_tei_text(_LEGACY_TEI2_XML) == "Shorter poems text."


def test_tei_header_fields_on_legacy_unnamespaced_tei2():
    from xml.etree import ElementTree

    root = ElementTree.fromstring(_LEGACY_TEI2_XML)
    assert ss._tei_author(root) == "Claudian"
    assert ss._tei_translator(root) == "Maurice Platnauer"
    assert ss._tei_publication_year(root) == 1922


def test_tei_author_and_editor_fields():
    from xml.etree import ElementTree

    root = ElementTree.fromstring(_LAT_TEI_XML)
    assert ss._tei_author(root) == "P. Vergilius Maro"
    assert ss._tei_plain_editor(root) == "J. B. Greenough"
    assert ss._tei_translator(root) is None
    assert ss._tei_publication_year(root) == 1881


def test_tei_plain_editor_accepts_explicit_role_editor_attribute():
    """Perseus doesn't always leave `role` off a non-translator editor --
    its own Beowulf glossary sets `role="editor"` explicitly (real case,
    caught live). `_tei_plain_editor` must accept both spellings.
    """
    from xml.etree import ElementTree

    xml = b"""<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
        <editor role="editor">Fr. Klaeber</editor>
    </titleStmt></fileDesc></teiHeader></TEI>"""
    assert ss._tei_plain_editor(ElementTree.fromstring(xml)) == "Fr. Klaeber"


def test_tei_translator_role_attribute():
    from xml.etree import ElementTree

    root = ElementTree.fromstring(_ENG_TEI_XML)
    assert ss._tei_translator(root) == "Theodore C. Williams"
    assert ss._tei_plain_editor(root) is None
    assert ss._tei_publication_year(root) == 1910


# --- free-text year/name fallback parsing ------------------------------------


def test_extract_year_takes_last_four_digit_match():
    assert ss._extract_year("Oxford: Clarendon Press, 1908-1920.") == 1920
    assert ss._extract_year("Boston: Houghton Mifflin, 1910.") == 1910
    assert ss._extract_year("no year here") is None


def test_extract_role_name_suffix_style():
    assert ss._extract_role_name("Greenough, J.B., editor. Boston: Ginn and Company, 1881.") == "Greenough, J.B."
    assert (
        ss._extract_role_name("Williams, Theodore, C, translator. Boston: Houghton Mifflin, 1910.")
        == "Williams, Theodore, C"
    )


def test_extract_role_name_prefix_style():
    assert ss._extract_role_name("Beowulf, edited by Fr. Klaeber, Boston: D. C. Heath, 1922.") == "Fr. Klaeber"
    assert (
        ss._extract_role_name("Beowulf, translated by James M. Garnett, Boston: Ginn & Company, 1912.")
        == "James M. Garnett"
    )


def test_extract_role_name_returns_none_when_no_role_word():
    assert ss._extract_role_name("Just some unrelated descriptive text.") is None


# --- AI-draft provenance -----------------------------------------------------


def test_ai_draft_provenance_extracts_model_name_and_year():
    label = "English (DeepSeek, 2026) using Brunetti analysis (AI draft)"
    assert ss._ai_draft_provenance(label, "") == ("DeepSeek", 2026)


def test_ai_draft_provenance_none_for_ordinary_translation():
    assert ss._ai_draft_provenance("English (James M. Garnett, 1912)", "translated by James M. Garnett") is None


# --- composition-year lookup -------------------------------------------------


def test_composition_year_from_llm(monkeypatch):
    monkeypatch.setattr(ss, "call_llm_json", lambda *a, **k: {"year": 19, "bc": True, "reasoning": "well-attested"})
    assert ss._composition_year("Aeneid", "Virgil", model=None) == (19, True)


def test_composition_year_none_when_genuinely_disputed(monkeypatch):
    monkeypatch.setattr(ss, "call_llm_json", lambda *a, **k: {"year": None, "bc": False, "reasoning": "disputed"})
    assert ss._composition_year("Disputed Work", "Anon", model=None) is None


def test_composition_year_raises_on_non_dict_llm_reply(monkeypatch):
    monkeypatch.setattr(ss, "call_llm_json", lambda *a, **k: "not a dict")
    with pytest.raises(ValueError):
        ss._composition_year("Aeneid", "Virgil", model=None)


def test_composition_year_retries_and_accepts_first_confident_answer(monkeypatch):
    """Caught live: the same (title, author) call can flip between a
    confident answer and "no year" across identical back-to-back calls
    for a borderline-but-knowable case. A later attempt succeeding should
    rescue the work rather than permanently skipping it over one unlucky
    call.
    """
    replies = iter(
        [
            {"year": None, "bc": False, "reasoning": "unsure this time"},
            {"year": 404, "bc": False, "reasoning": "confident this time"},
        ]
    )
    monkeypatch.setattr(ss, "call_llm_json", lambda *a, **k: next(replies))
    assert ss._composition_year("Carminum minorum corpusculum", "Claudian", model=None) == (404, False)


def test_composition_year_none_after_exhausting_every_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ss,
        "call_llm_json",
        lambda *a, **k: calls.append(1) or {"year": None, "bc": False, "reasoning": "disputed"},
    )
    assert ss._composition_year("Beowulf", "unknown", model=None) is None
    assert len(calls) == ss._COMPOSITION_YEAR_ATTEMPTS


# --- end-to-end resolver ------------------------------------------------------


def test_resolve_structured_documents_returns_none_for_unregistered_url():
    assert ss.resolve_structured_documents("https://www.gutenberg.org/cache/epub/1/pg1.txt") is None


def test_resolve_structured_documents_resolves_edition_and_translation(monkeypatch):
    cts_url = "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/__cts__.xml"
    lat_url = (
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/"
        "phi0690.phi003.perseus-lat2.xml"
    )
    eng_url = (
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/"
        "phi0690.phi003.perseus-eng2.xml"
    )
    routes = {
        cts_url: _response(200, content=_CTS_XML),
        lat_url: _response(200, content=_LAT_TEI_XML),
        eng_url: _response(200, content=_ENG_TEI_XML),
    }
    monkeypatch.setattr(ss.httpx, "get", _fake_httpx_get(routes))
    monkeypatch.setattr(ss, "_composition_year", lambda title, author, model: (19, True))

    documents = ss.resolve_structured_documents(cts_url)
    assert documents is not None
    by_role = {d.role: d for d in documents}
    assert set(by_role) == {"edition", "translation"}  # commentary dropped

    edition = by_role["edition"]
    assert edition.lang_code == "lat"
    assert edition.year == -19  # 19 BC, from the mocked composition-year lookup, not the 1881 edition date
    assert edition.author == "P. Vergilius Maro"
    assert "Arma virumque cano" in edition.content

    translation = by_role["translation"]
    assert translation.lang_code == "eng"
    assert translation.year == 1910  # the translation's own real publication year, no lookup needed
    assert translation.author == "Theodore C. Williams"


def test_resolve_structured_documents_skips_edition_with_no_confident_year(monkeypatch):
    cts_url = "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/__cts__.xml"
    lat_url = (
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/"
        "phi0690.phi003.perseus-lat2.xml"
    )
    eng_url = (
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/"
        "phi0690.phi003.perseus-eng2.xml"
    )
    routes = {
        cts_url: _response(200, content=_CTS_XML),
        lat_url: _response(200, content=_LAT_TEI_XML),
        eng_url: _response(200, content=_ENG_TEI_XML),
    }
    monkeypatch.setattr(ss.httpx, "get", _fake_httpx_get(routes))
    monkeypatch.setattr(ss, "_composition_year", lambda title, author, model: None)

    documents = ss.resolve_structured_documents(cts_url)
    assert documents is not None
    roles = {d.role for d in documents}
    assert roles == {"translation"}  # the edition was skipped; the translation didn't need the lookup at all


_GLOSSARY_CTS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ti:work xmlns:ti="http://chs.harvard.edu/xmlns/cts" urn="urn:cts:angLit:anon.klaeber_glossary" xml:lang="ang">
    <ti:title xml:lang="eng">Klaeber's Beowulf Glossary</ti:title>
    <ti:edition urn="urn:cts:angLit:anon.klaeber_glossary.perseus-mul1" xml:lang="eng">
        <ti:label xml:lang="eng">Old English-English glossary (Fr. Klaeber, 1922)</ti:label>
        <ti:description xml:lang="eng">The glossary to Klaeber's 1922 edition of Beowulf; 3,784 entries.</ti:description>
    </ti:edition>
</ti:work>
"""

_GLOSSARY_TEI_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Klaeber's Beowulf Glossary</title><editor role="editor">Fr. Klaeber</editor></titleStmt>
      <sourceDesc>
        <biblStruct><monogr><imprint><date>1922</date></imprint></monogr></biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text xml:lang="eng"><body><div type="edition"><p>abeodan to announce</p></div></body></text>
</TEI>
"""


def test_resolve_structured_documents_skips_composition_lookup_for_edition_in_a_different_language(monkeypatch):
    """A <ti:edition> entry whose own xml:lang differs from the work's declared original
    language (real case: Beowulf's glossary is tagged edition/xml:lang="eng" even though the
    work's own original language is "ang") is a modern reference work with its own real print
    date already in its metadata -- it must use that directly, never the composition-year
    lookup meant for an actual ancient-language original.
    """
    cts_url = (
        "https://raw.githubusercontent.com/PerseusDL/canonical-angLit/master/data/anon/klaeber_glossary/__cts__.xml"
    )
    doc_url = (
        "https://raw.githubusercontent.com/PerseusDL/canonical-angLit/master/data/anon/klaeber_glossary/"
        "anon.klaeber_glossary.perseus-mul1.xml"
    )
    routes = {cts_url: _response(200, content=_GLOSSARY_CTS_XML), doc_url: _response(200, content=_GLOSSARY_TEI_XML)}
    monkeypatch.setattr(ss.httpx, "get", _fake_httpx_get(routes))

    def _fail_if_called(*a: object, **k: object) -> None:
        raise AssertionError("composition-year lookup must not run for an edition outside the work's own language")

    monkeypatch.setattr(ss, "_composition_year", _fail_if_called)

    (document,) = ss.resolve_structured_documents(cts_url)  # type: ignore[misc]
    assert document.year == 1922  # the glossary's own real print date, from its teiHeader
    assert document.author == "Fr. Klaeber"


# --- discovery ----------------------------------------------------------------


def test_discover_perseus_cts_urls_filters_to_work_level_only(monkeypatch):
    tree = {
        "truncated": False,
        "tree": [
            {"path": "data/phi0690/__cts__.xml", "type": "blob"},  # author-level, excluded
            {"path": "data/phi0690/phi003/__cts__.xml", "type": "blob"},  # work-level, included
            {"path": "data/phi0690/phi003/phi0690.phi003.perseus-lat2.xml", "type": "blob"},  # a text file, excluded
            {"path": "data/phi0690/phi003", "type": "tree"},  # a directory, excluded
        ],
    }
    monkeypatch.setattr(ss.httpx, "get", lambda *a, **k: _response(200, json=tree))

    urls = ss.discover_perseus_cts_urls("PerseusDL/canonical-latinLit")
    assert urls == [
        "https://raw.githubusercontent.com/PerseusDL/canonical-latinLit/master/data/phi0690/phi003/__cts__.xml"
    ]


def test_discover_perseus_cts_urls_raises_on_truncated_tree(monkeypatch):
    monkeypatch.setattr(ss.httpx, "get", lambda *a, **k: _response(200, json={"truncated": True, "tree": []}))
    with pytest.raises(ValueError):
        ss.discover_perseus_cts_urls("PerseusDL/canonical-latinLit")
