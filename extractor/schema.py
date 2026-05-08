"""Build the empty target metadata template.

Mirrors data/example/metadata.json so the pipeline can fill in
fields it can identify and leave the rest as empty strings/lists.
"""
from __future__ import annotations

from typing import Any, Dict


def empty_metadata() -> Dict[str, Any]:
    return {
        "general": {
            "uid": "",
            "issue": "",
            "volume": "",
            "keywords": [],
            "lastPage": "",
            "firstPage": "",
            "subtitle": "",
            "articleId": "",
            "issnPrint": "",
            "issueYear": "",
            "articleUrl": "",
            "categories": [],
            "issnOnline": "",
            "issueTitle": "",
            "articleType": "research-article",
            "articleTitle": "",
            "journalTitle": "",
            "abbrevJournalTitle": "",
            "abstract": "",
        },
        "dateForm": {
            "revisedDate": {},
            "acceptedDate": {},
            "receivedDate": {},
            "publicationDate": {},
            "issuePublicationDate": {},
        },
        "authorForm": [],
        "editorForm": [],
        "reviewerForm": [],
        "publisherForm": {
            "name": "",
            "location": "",
        },
        "permissionForm": {
            "licenseUrl": "",
            "copyrightYear": "",
            "copyrightHolder": "",
            "openAccessLicense": False,
            "copyrightStatement": "",
            "licenseInformation": "",
        },
    }
