"""Flask web app for the Journal Metadata Extractor.

Routes:
    GET  /            — upload form
    POST /extract     — accept a PDF, run the pipeline, render result page
    POST /api/extract — JSON API (returns metadata JSON)
"""
from __future__ import annotations

import json
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from extractor import extract_metadata

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, "data", "uploads")
RESULT_DIR = os.path.join(ROOT, "data", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ALLOWED_EXT = {".pdf"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def _is_allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    if "pdf" not in request.files:
        return render_template("index.html", error="No file uploaded."), 400
    f = request.files["pdf"]
    if not f.filename:
        return render_template("index.html", error="No file selected."), 400
    if not _is_allowed(f.filename):
        return render_template("index.html", error="Only PDF files are allowed."), 400

    name = secure_filename(f.filename)
    job_id = uuid.uuid4().hex[:12]
    saved_pdf = os.path.join(UPLOAD_DIR, f"{job_id}_{name}")
    f.save(saved_pdf)

    try:
        metadata = extract_metadata(saved_pdf)
    except Exception as exc:  # pragma: no cover
        return render_template("index.html", error=f"Extraction failed: {exc}"), 500

    result_path = os.path.join(RESULT_DIR, f"{job_id}.json")
    with open(result_path, "w", encoding="utf-8") as out:
        json.dump(metadata, out, indent=2, ensure_ascii=False)

    return render_template(
        "result.html",
        metadata=metadata,
        metadata_json=json.dumps(metadata, indent=2, ensure_ascii=False),
        filename=name,
        job_id=job_id,
    )


@app.route("/download/<job_id>")
def download(job_id: str):
    path = os.path.join(RESULT_DIR, f"{job_id}.json")
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, as_attachment=True,
                     download_name=f"metadata_{job_id}.json",
                     mimetype="application/json")


@app.route("/api/extract", methods=["POST"])
def api_extract():
    if "pdf" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["pdf"]
    if not f.filename or not _is_allowed(f.filename):
        return jsonify({"error": "invalid file"}), 400
    name = secure_filename(f.filename)
    job_id = uuid.uuid4().hex[:12]
    saved_pdf = os.path.join(UPLOAD_DIR, f"{job_id}_{name}")
    f.save(saved_pdf)
    try:
        metadata = extract_metadata(saved_pdf)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(metadata)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
