"""Classifies tokens as name / common word / anachronism — spec §2, §3b.

Names bypass sense-retrieval entirely; common words go to sense retrieval.
Falls back to a contributor-supplied `tokens[].class` hint when present.
Not yet implemented.
"""
