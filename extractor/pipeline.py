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


def _split_authors(author_lines: List[str]) -> List[str]:
    """Split a line like 'Def Reinhard*1, Avian Renohardian2' into individual names."""
    names: List[str] = []
    for line in author_lines:
        line = re.sub(r"[\*†‡§¶]", "", line)
        line = re.sub(r"(?<=[A-Za-z])\d+", "", line)  # strip superscript digits
        for part in re.split(r",|\band\b|;|&", line, flags=re.IGNORECASE):
            p = part.strip(" .;:-")
            if p and 2 <= len(p.split()) <= 6 and not any(c.isdigit() for c in p):
                names.append(p)
    # dedupe preserving order
    seen = set()
    out = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


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


def _detect_publisher(text: str, ner_orgs: List[str]) -> Dict[str, str]:
    m = re.search(
        r"Published\s+by\s+([^\n.]{3,80})",
        text, re.IGNORECASE,
    )
    name = ""
    if m:
        name = m.group(1).strip(" .,;:-")
    elif ner_orgs:
        # pick an ORG that looks publisher-like (contains University/Press/Publisher)
        for o in ner_orgs:
            if re.search(r"University|Press|Publisher|Penerbit|Universitas", o, re.IGNORECASE):
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
        g["issueYear"] = str(dates["publicationDate"]["year"])

    # authors: combine CRF + spaCy PERSON
    crf_authors = _split_authors(crf_groups.get("AUTHOR", []))
    persons = ner.get("PERSON", [])
    seen = {a.lower() for a in crf_authors}
    for p in persons:
        if p.lower() not in seen and 2 <= len(p.split()) <= 5:
            crf_authors.append(p)
            seen.add(p.lower())

    affil_lines = crf_groups.get("AFFIL", [])
    emails = rule.get("emails") or []
    orcids = rule.get("orcids") or []

    authors = []
    for i, name in enumerate(crf_authors[:10], start=1):
        record = _author_record(
            idx=i + 1,  # match example which starts at author-2
            name=name,
            email=emails[i - 1] if i - 1 < len(emails) else "",
            orcid=orcids[i - 1] if i - 1 < len(orcids) else "",
            affiliations=[affil_lines[i - 1]] if i - 1 < len(affil_lines)
                         else (affil_lines[:1] if affil_lines else []),
        )
        if i == 1 and emails:
            record["corresp"] = True
        # try to fill country from spaCy GPE
        for gpe in ner.get("GPE", []):
            if record["affiliations"] and gpe in record["affiliations"][0]:
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
    out["publisherForm"] = _detect_publisher(text, ner.get("ORG", []))
    # publisher location guess: first GPE
    if not out["publisherForm"]["location"] and ner.get("GPE"):
        out["publisherForm"]["location"] = ner["GPE"][0]

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
