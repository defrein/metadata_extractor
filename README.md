# Journal Metadata Extractor

Hybrid extraction of scientific journal article metadata from PDF files,
combining **rule-based regex**, a **CRF sequence model**, and **spaCy NER**.
Web UI built with **Flask + Tailwind**.

## Features

- **PyMuPDF** layout-aware text extraction (font size, position, bold)
- **Rule-based regex** for: DOI, ISSN (print/online), volume, issue, page
  range, emails, ORCID, dates (received / revised / accepted / published),
  copyright statement, license URL, keywords, abstract
- **CRF (sklearn-crfsuite)** for token-level sequence labeling of
  `TITLE / AUTHOR / AFFIL / JOURNAL / ABSTRACT / KEYWORD / OTHER` with
  layout-aware features (relative font size, bold, position)
- **spaCy NER** to enrich authors (PERSON), publisher (ORG), country (GPE)
- **Hybrid pipeline** that merges all three into the JSON schema in
  `data/example/metadata.json`
- **Flask UI** — drag/click to upload, instant rendered result page,
  download as JSON, plus a JSON API at `POST /api/extract`

## Project layout

```
app.py                  Flask web app
extractor/
  pdf_reader.py         PyMuPDF extraction
  rule_based.py         Regex extractors
  crf_extractor.py      CRF model + features (with heuristic fallback)
  spacy_ner.py          spaCy NER wrapper
  pipeline.py           Hybrid orchestrator → JSON schema
  schema.py             Empty-template builder
train_crf.py            Trains models/crf_model.pkl from training samples
templates/              base.html · index.html · result.html (Tailwind via CDN)
data/
  example/metadata.json Target schema reference
  training/samples.json Labeled token/label sequences
  uploads/              Uploaded PDFs (created at runtime)
  results/              Saved JSON results (created at runtime)
models/crf_model.pkl    Pickled CRF model (created by train_crf.py)
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_crf.py
python app.py
```

Then open <http://127.0.0.1:5000>.

> The CRF model is optional — without it, `crf_extractor.py` falls back
> to a layout-only heuristic (largest-font line on page 0 = title, etc.),
> so the pipeline still works end-to-end.

## API

```bash
curl -F "pdf=@article.pdf" http://127.0.0.1:5000/api/extract
```

Returns the same JSON shape as `data/example/metadata.json`.

## How fields are sourced

| Field                                       | Source                              |
|---------------------------------------------|-------------------------------------|
| `articleId`, `articleUrl`                   | regex (DOI)                         |
| `issnPrint`, `issnOnline`                   | regex (labeled context)             |
| `volume`, `issue`, `firstPage`, `lastPage`  | regex                               |
| `keywords`, `abstract`                      | regex (labeled blocks)              |
| `articleTitle`, `journalTitle`              | CRF + layout fallback               |
| `authorForm[*].name`                        | CRF + spaCy PERSON                  |
| `authorForm[*].affiliations`                | CRF AFFIL                           |
| `authorForm[*].email/orcid`                 | regex                               |
| `authorForm[*].country`                     | spaCy GPE matched against affil     |
| `publisherForm.name/location`               | regex + spaCy ORG/GPE               |
| `dateForm.*`                                | regex (received/revised/etc.)       |
| `permissionForm.*`                          | regex (copyright/license)           |
| `general.uid`                               | generated UUID                      |

## Extending the CRF

Add more labeled samples to `data/training/samples.json` (each sample is
`{tokens, labels, layout.size_per_token}`), then re-run
`python train_crf.py`. The richer the layout cues, the better the model.
