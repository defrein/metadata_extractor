"""Hybrid pipeline: combines rule-based, CRF, and spaCy outputs into JSON."""
from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List, Optional

from . import rule_based, spacy_ner
from .crf_extractor import CRFExtractor
from .pdf_reader import read_pdf
from .schema import empty_metadata

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "crf_model.pkl",
)


def _clean_title(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(" .,:;-")


_AFFIL_KEYWORDS = re.compile(
    r"\b(Fakultas|Universitas|Department|Departement|University|"
    r"Institute|Institut|Faculty|Sekolah\s+Tinggi|Politeknik|Akademi|"
    r"Jl\.|Jalan\s|Kampus|Indonesia|USA|UK)\b",
    re.IGNORECASE,
)


def _split_authors(author_lines: List[str]) -> List[Tuple[str, Optional[int]]]:
    """Split author lines into (name, super_index_or_None) tuples.

    Handles 'Author One¹, Author Two², Author Three³' (single line) and
    multi-line variants. Superscript digits are captured for email matching.
    """
    out: List[Tuple[str, Optional[int]]] = []
    seen: set = set()
    for line in author_lines:
        if _AFFIL_KEYWORDS.search(line):
            continue
        # split by separators
        parts = re.split(r",|\band\b|;|&", line, flags=re.IGNORECASE)
        for part in parts:
            p = re.sub(r"[\*†‡§¶]", "", part).strip(" .;:-")
            if not p:
                continue
            # extract trailing/leading superscript digit
            super_idx: Optional[int] = None
            m = re.search(r"(\d+)\s*$", p)
            if m:
                super_idx = int(m.group(1))
                p = p[: m.start()].strip()
            else:
                m = re.match(r"^(\d+)\s*", p)
                if m:
                    super_idx = int(m.group(1))
                    p = p[m.end():].strip()
            # strip stray inner digits
            p = re.sub(r"(?<=[A-Za-z])\d+", "", p).strip(" .,")
            tokens = p.split()
            if not (2 <= len(tokens) <= 5):
                continue
            if not all(t[0].isupper() for t in tokens if t[0].isalpha()):
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((p, super_idx))
    return out


def _split_affiliations(affil_lines: List[str]) -> List[str]:
    """Group an affiliation block into one logical institution string.

    Most non-multi-affil papers have a single shared affiliation that should
    be assigned to all authors.
    """
    if not affil_lines:
        return []
    joined = " ".join(affil_lines)
    joined = re.sub(r"\s+", " ", joined).strip(" ,;.")
    return [joined] if joined else []


def _author_record(idx: int, name: str, email: str = "", orcid: str = "",
                   affiliations: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "id": f"author-{idx}",
        "name": name,
        "email": email,
        "orcid": orcid,
        "phone": "",
        "scopus": "",
        "corresp": False,
        "country": "",
        "affiliations": affiliations or [],
    }


_DOMAIN_TO_PUBLISHER = {
    "unbaja.ac.id": "Universitas Banten Jaya",
    "ui.ac.id": "Universitas Indonesia",
    "ugm.ac.id": "Universitas Gadjah Mada",
    "itb.ac.id": "Institut Teknologi Bandung",
    "unair.ac.id": "Universitas Airlangga",
    "ipb.ac.id": "IPB University",
    "its.ac.id": "Institut Teknologi Sepuluh Nopember",
    "unhas.ac.id": "Universitas Hasanuddin",
}


def _publisher_from_emails(emails: List[str]) -> str:
    for e in emails:
        domain = e.split("@", 1)[-1].lower()
        if domain in _DOMAIN_TO_PUBLISHER:
            return _DOMAIN_TO_PUBLISHER[domain]
    return ""


def _detect_publisher(text: str, ner_orgs: List[str],
                      affiliations: List[str], emails: List[str]) -> Dict[str, str]:
    m = re.search(r"Published\s+by\s+([^\n.]{3,80})", text, re.IGNORECASE)
    name = ""
    if m:
        name = m.group(1).strip(" .,;:-")
    if not name:
        # try to pull a "Universitas X" / "University of X" out of affiliations
        for a in affiliations:
            mm = re.search(
                r"(Universitas\s+[A-Z][\w\s]+?|University\s+of\s+[A-Z][\w\s]+?|"
                r"Institut\s+Teknologi\s+[A-Z][\w\s]+?|Politeknik\s+[A-Z][\w\s]+?)"
                r"(?:,|\.|$)",
                a,
            )
            if mm:
                name = mm.group(1).strip(" ,.;")
                break
    if not name:
        name = _publisher_from_emails(emails)
    if not name and ner_orgs:
        for o in ner_orgs:
            if re.search(r"University|Press|Publisher|Penerbit|Universitas",
                         o, re.IGNORECASE):
                name = o
                break
    return {"name": name, "location": ""}


def _detect_journal_title(crf_groups: Dict[str, List[str]],
                          rule_text: str) -> str:
    if crf_groups.get("JOURNAL"):
        return _clean_title(crf_groups["JOURNAL"][0])
    return ""


def _detect_article_title(crf_groups: Dict[str, List[str]]) -> str:
    if crf_groups.get("TITLE"):
        return _clean_title(" ".join(crf_groups["TITLE"]))
    return ""


def _detect_categories(text: str) -> List[Dict[str, Any]]:
    m = re.search(
        r"(?:Subject|Subject\s+Area|Bidang)\s*[:\-]\s*([^\n]{3,80})",
        text, re.IGNORECASE,
    )
    if not m:
        return []
    return [{"subject": m.group(1).strip(" .;:-"), "subjectGroup": []}]


def extract_metadata(pdf_path: str,
                     model_path: Optional[str] = None) -> Dict[str, Any]:
    """Main entry point: PDF path → metadata dict matching the example schema."""
    doc = read_pdf(pdf_path, max_pages=4)
    text = doc.text_of_first_n_pages(3)

    rule = rule_based.extract_rule_based(text)
    crf = CRFExtractor(model_path or DEFAULT_MODEL_PATH)
    crf_groups = crf.predict(doc)
    ner = spacy_ner.extract_entities(text)

    out = empty_metadata()
    g = out["general"]

    g["uid"] = str(uuid.uuid4())
    g["articleId"] = rule.get("articleId") or ""
    g["articleUrl"] = rule.get("articleUrl") or ""
    g["issnPrint"] = rule.get("issnPrint") or ""
    g["issnOnline"] = rule.get("issnOnline") or ""
    g["volume"] = rule.get("volume") or ""
    g["issue"] = rule.get("issue") or ""
    g["issueYear"] = rule.get("issueYear") or ""
    g["firstPage"] = rule.get("firstPage") or ""
    g["lastPage"] = rule.get("lastPage") or ""
    g["keywords"] = rule.get("keywords") or []
    g["abstract"] = rule.get("abstract") or ""
    g["categories"] = _detect_categories(text)

    g["articleTitle"] = _detect_article_title(crf_groups)
    g["journalTitle"] = _detect_journal_title(crf_groups, text)
    if g["journalTitle"]:
        # crude abbreviation: take initials
        g["abbrevJournalTitle"] = "".join(
            w[0] for w in g["journalTitle"].split() if w[:1].isupper()
        )[:8]

    # dates
    dates = rule.get("dates") or {}
    for k in ("receivedDate", "revisedDate", "acceptedDate", "publicationDate"):
        if dates.get(k):
            out["dateForm"][k] = {**dates[k]}
    if dates.get("publicationDate"):
        out["dateForm"]["publicationDate"]["publicationFormat"] = "electronic"
        out["dateForm"]["issuePublicationDate"] = {
            **dates["publicationDate"],
            "publicationFormat": "electronic",
        }
        if not g["issueYear"]:
            g["issueYear"] = str(dates["publicationDate"]["year"])

    # authors: prefer CRF/heuristic (with superscript); fall back to spaCy
    crf_author_pairs = _split_authors(crf_groups.get("AUTHOR", []))
    if not crf_author_pairs:
        # fall back to spaCy PERSON only when CRF gave us nothing
        persons = ner.get("PERSON", [])
        for p in persons[:6]:
            if 2 <= len(p.split()) <= 4 and not _AFFIL_KEYWORDS.search(p):
                crf_author_pairs.append((p, None))

    affil_lines = _split_affiliations(crf_groups.get("AFFIL", []))
    emails_super = rule.get("emailsBySuper") or []
    emails = rule.get("emails") or []
    orcids = rule.get("orcids") or []

    # Map superscript index → email
    email_by_super = {idx: addr for idx, addr in emails_super if idx is not None}

    authors = []
    for i, (name, super_idx) in enumerate(crf_author_pairs[:10], start=1):
        # match email by superscript number when available, else by position
        email = ""
        if super_idx is not None and super_idx in email_by_super:
            email = email_by_super[super_idx]
        elif i - 1 < len(emails):
            email = emails[i - 1]
        # affiliation: share single block across all authors when only one
        affil = affil_lines[:1] if affil_lines else []
        record = _author_record(
            idx=i + 1,
            name=name,
            email=email,
            orcid=orcids[i - 1] if i - 1 < len(orcids) else "",
            affiliations=affil,
        )
        if i == 1 and email:
            record["corresp"] = True
        for gpe in ner.get("GPE", []):
            if affil and gpe.lower() in affil[0].lower() and len(gpe) > 3:
                record["country"] = gpe
                break
        authors.append(record)
    out["authorForm"] = authors

    # placeholder editor/reviewer entries
    out["editorForm"] = [{
        "id": "editor-1", "name": "", "email": "", "orcid": "",
        "phone": "", "scopus": "", "country": "", "affiliations": [],
    }]
    out["reviewerForm"] = [{
        "id": "reviewer-1", "name": "", "email": "", "orcid": "",
        "phone": "", "scopus": "", "country": "", "affiliations": [],
    }]

    # publisher
    out["publisherForm"] = _detect_publisher(
        text, ner.get("ORG", []), affil_lines, emails,
    )
    # publisher location: prefer a GPE that actually appears in the affiliation
    if not out["publisherForm"]["location"]:
        for gpe in ner.get("GPE", []):
            if affil_lines and gpe.lower() in affil_lines[0].lower():
                out["publisherForm"]["location"] = gpe
                break

    # license + copyright
    cp = rule.get("copyright") or {}
    lic = rule.get("license") or {}
    perm = out["permissionForm"]
    perm["copyrightYear"] = cp.get("copyrightYear", "")
    perm["copyrightHolder"] = cp.get("copyrightHolder", "")
    perm["copyrightStatement"] = cp.get("copyrightStatement", "")
    perm["licenseUrl"] = lic.get("licenseUrl", "")
    perm["openAccessLicense"] = bool(lic.get("openAccessLicense", False))
    perm["licenseInformation"] = lic.get("licenseInformation", "")

    return out
