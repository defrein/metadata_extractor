"""Rule-based regex extractors for deterministic metadata fields.

Handles fields that have stable surface forms: DOI, ISSN, emails, ORCID,
dates, page ranges, volume/issue numbers, copyright, license info.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
ISSN_RE = re.compile(r"\b(\d{4}-\d{3}[\dXx])\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
ORCID_RE = re.compile(r"\b(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])\b")
URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)

VOLUME_RE = re.compile(r"\b(?:Vol(?:ume)?\.?|Vol)\s*[:\.]?\s*(\d{1,4})", re.IGNORECASE)
ISSUE_RE = re.compile(r"\b(?:Issue|No\.?|Number|Nomor)\s*[:\.]?\s*(\d{1,4})", re.IGNORECASE)
PAGES_RE = re.compile(
    r"\b(?:pp?\.?|pages?|halaman)\s*[:\.]?\s*(\d{1,5})\s*[-‐-―−]\s*(\d{1,5})",
    re.IGNORECASE,
)
PAGES_PLAIN_RE = re.compile(r"\b(\d{1,5})\s*[-‐-―−]\s*(\d{1,5})\b")

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
COPYRIGHT_RE = re.compile(
    r"(?:Copyright|©|\(c\))\s*©?\s*(\d{4})\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

LICENSE_URL_RE = re.compile(
    r"https?://(?:creativecommons\.org|opensource\.org)/[^\s)\]]+",
    re.IGNORECASE,
)

MONTHS_EN = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
MONTHS_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5,
    "juni": 6, "juli": 7, "agustus": 8, "september": 9, "oktober": 10,
    "november": 11, "desember": 12,
}
MONTH_PATTERN = "|".join(sorted(set(list(MONTHS_EN.keys()) + list(MONTHS_ID.keys())),
                                 key=len, reverse=True))

DATE_TEXT_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
DATE_NUM_RE = re.compile(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})\b")

RECEIVED_RE = re.compile(
    r"(?:Received|Diterima|Submitted)\s*[:\-]?\s*([^\n;]{4,40})",
    re.IGNORECASE,
)
REVISED_RE = re.compile(
    r"(?:Revised|Direvisi)\s*[:\-]?\s*([^\n;]{4,40})",
    re.IGNORECASE,
)
ACCEPTED_RE = re.compile(
    r"(?:Accepted|Diterima\s+untuk\s+publikasi|Disetujui)\s*[:\-]?\s*([^\n;]{4,40})",
    re.IGNORECASE,
)
PUBLISHED_RE = re.compile(
    r"(?:Published|Dipublikasikan|Publication\s+date)\s*[:\-]?\s*([^\n;]{4,40})",
    re.IGNORECASE,
)

KEYWORDS_BLOCK_RE = re.compile(
    r"(?:Keywords?|Kata\s+kunci|Indexing\s+terms)\s*[:\-]\s*(.+?)(?:\n\s*\n|\n[A-Z][^\n]{0,30}:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

ABSTRACT_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*Abstract(?:\s*[:\-])?\s*(.+?)(?:\n\s*Keywords?\s*[:\-]|\n\s*Kata\s+kunci\s*[:\-]|\n\s*1\.?\s+Introduction|\n\s*Pendahuluan|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def parse_date_text(text: str) -> Optional[Dict[str, int]]:
    """Parse a free-form date string into {day, month, year}."""
    text = text.strip().rstrip(".,;:")
    m = DATE_TEXT_RE.search(text)
    if m:
        day = int(m.group(1))
        mon_key = m.group(2).lower()
        month = MONTHS_EN.get(mon_key) or MONTHS_ID.get(mon_key)
        year = int(m.group(3))
        if month:
            return {"day": day, "month": month, "year": year}
    m = DATE_NUM_RE.search(text)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # heuristic: if a > 12, it's day-month-year; else assume day-month-year too (id format)
        day, month = (a, b) if a <= 31 and b <= 12 else (b, a)
        return {"day": day, "month": month, "year": y}
    return None


def find_doi(text: str) -> Optional[str]:
    m = DOI_RE.search(text)
    return m.group(1) if m else None


def find_issns(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (issn_print, issn_online) using context labels when present."""
    issn_print = None
    issn_online = None
    for m in re.finditer(
        r"(p[\-\s]?ISSN|ISSN\s*\(\s*Print\s*\)|Print\s*ISSN)\s*[:\-]?\s*(\d{4}-\d{3}[\dXx])",
        text, re.IGNORECASE,
    ):
        issn_print = m.group(2)
        break
    for m in re.finditer(
        r"(e[\-\s]?ISSN|ISSN\s*\(\s*Online\s*\)|Online\s*ISSN|Electronic\s*ISSN)\s*[:\-]?\s*(\d{4}-\d{3}[\dXx])",
        text, re.IGNORECASE,
    ):
        issn_online = m.group(2)
        break
    if not issn_print and not issn_online:
        all_issns = ISSN_RE.findall(text)
        if all_issns:
            issn_print = all_issns[0]
            if len(all_issns) > 1:
                issn_online = all_issns[1]
    return issn_print, issn_online


def find_emails(text: str) -> List[str]:
    seen = []
    for e in EMAIL_RE.findall(text):
        if e not in seen:
            seen.append(e)
    return seen


def find_orcids(text: str) -> List[str]:
    seen = []
    for o in ORCID_RE.findall(text):
        if o not in seen:
            seen.append(o)
    return seen


def find_volume(text: str) -> Optional[str]:
    m = VOLUME_RE.search(text)
    return m.group(1) if m else None


def find_issue(text: str) -> Optional[str]:
    m = ISSUE_RE.search(text)
    return m.group(1) if m else None


def find_pages(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = PAGES_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    # Indonesian/general header pattern: "...Tahun 2023 E-ISSN: xxxx-xxxx 102"
    # or "Vol 5 No 2 ... 102" — capture the trailing 1-4 digit page number
    m = re.search(
        r"(?:Tahun\s+\d{4}|Year\s+\d{4})\b[^\n]{0,80}?\b(\d{1,4})\s*$",
        text, re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return m.group(1), None
    m = re.search(
        r"E[-\s]?ISSN\s*[:\-]?\s*\d{4}-\d{3}[\dXx]\s+(\d{1,4})\b",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1), None
    return None, None


def find_issue_year(text: str) -> Optional[str]:
    """Issue year from 'Tahun YYYY' / 'Year YYYY' / '(YYYY)'."""
    m = re.search(r"\b(?:Tahun|Year)\s+(\d{4})\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\((19|20)(\d{2})\)", text)
    if m:
        return m.group(1) + m.group(2)
    return None


def clean_email(addr: str) -> str:
    """Strip leading superscript digits/markers from an email local part."""
    return re.sub(r"^[\d\*†‡§¶]+", "", addr).strip()


def parse_email_supers(text: str) -> List[Tuple[Optional[int], str]]:
    """Return list of (superscript_index_or_None, cleaned_email)."""
    out: List[Tuple[Optional[int], str]] = []
    seen = set()
    for raw in EMAIL_RE.findall(text):
        m = re.match(r"^(\d+)(.+)$", raw)
        if m:
            idx = int(m.group(1))
            addr = m.group(2)
        else:
            idx, addr = None, raw
        if addr.lower() in seen:
            continue
        seen.add(addr.lower())
        out.append((idx, addr))
    return out


def find_keywords(text: str) -> List[str]:
    m = KEYWORDS_BLOCK_RE.search(text)
    if not m:
        return []
    block = m.group(1).strip()
    block = re.sub(r"\s+", " ", block)
    parts = re.split(r"[;,•·•]|\s{2,}", block)
    out = []
    for p in parts:
        p = p.strip(" .;:-")
        if 2 <= len(p) <= 80 and p:
            out.append(p)
    return out[:15]


def find_abstract(text: str) -> Optional[str]:
    m = ABSTRACT_BLOCK_RE.search(text)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip()


def find_copyright(text: str) -> Dict[str, str]:
    m = COPYRIGHT_RE.search(text)
    if not m:
        return {}
    year = m.group(1)
    holder = m.group(2).strip()
    holder = re.sub(r"\s+", " ", holder)[:120]
    return {
        "copyrightYear": year,
        "copyrightHolder": holder,
        "copyrightStatement": f"Copyright © {year} {holder}.",
    }


def find_license(text: str) -> Dict[str, object]:
    m = LICENSE_URL_RE.search(text)
    out: Dict[str, object] = {}
    if m:
        out["licenseUrl"] = m.group(0).rstrip(".,)")
        out["openAccessLicense"] = True
    # license information sentence
    li = re.search(
        r"(This is an Open Access article[^\n]+(?:\n[^\n]+){0,3})",
        text, re.IGNORECASE,
    )
    if li:
        out["licenseInformation"] = re.sub(r"\s+", " ", li.group(1)).strip()
    return out


def find_dates(text: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    mappings = [
        ("receivedDate", RECEIVED_RE),
        ("revisedDate", REVISED_RE),
        ("acceptedDate", ACCEPTED_RE),
        ("publicationDate", PUBLISHED_RE),
    ]
    for key, regex in mappings:
        m = regex.search(text)
        if m:
            d = parse_date_text(m.group(1))
            if d:
                out[key] = d
    return out


def extract_rule_based(text: str) -> Dict[str, object]:
    """Run all regex extractors on the raw text and return a normalized dict."""
    doi = find_doi(text)
    issn_print, issn_online = find_issns(text)
    first_page, last_page = find_pages(text)
    email_supers = parse_email_supers(text)
    emails = [addr for _, addr in email_supers]
    orcids = find_orcids(text)
    keywords = find_keywords(text)
    abstract = find_abstract(text)
    dates = find_dates(text)
    copyright_data = find_copyright(text)
    license_data = find_license(text)

    result: Dict[str, object] = {
        "doi": doi,
        "articleId": doi or "",
        "articleUrl": f"https://doi.org/{doi}" if doi else "",
        "issnPrint": issn_print or "",
        "issnOnline": issn_online or "",
        "volume": find_volume(text) or "",
        "issue": find_issue(text) or "",
        "issueYear": find_issue_year(text) or "",
        "firstPage": first_page or "",
        "lastPage": last_page or "",
        "emails": emails,
        "emailsBySuper": email_supers,
        "orcids": orcids,
        "keywords": keywords,
        "abstract": abstract or "",
        "dates": dates,
        "copyright": copyright_data,
        "license": license_data,
    }
    return result
