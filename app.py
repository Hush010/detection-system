from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
import os

from flask import Flask, jsonify, send_file, request
from detector import analyze_text
import PyPDF2

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Unable to read PDF: {str(e)}. The file may be corrupted or invalid.")
    except Exception as e:
        raise ValueError(f"PDF parsing failed: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            xml_data = archive.read("word/document.xml")

        root = ET.fromstring(xml_data)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in root.findall(".//w:p", ns):
            texts = [node.text for node in paragraph.findall(".//w:t", ns) if node.text]
            if texts:
                paragraphs.append("".join(texts))

        return "\n".join(paragraphs)
    except zipfile.BadZipFile:
        raise ValueError("Unable to read DOCX: The file is corrupted or not a valid Office document.")
    except KeyError:
        raise ValueError("Unable to read DOCX: Missing document.xml. The file may be corrupted.")
    except Exception as e:
        raise ValueError(f"DOCX parsing failed: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def extract_text_from_file(file_storage) -> str:
    filename = (file_storage.filename or "").lower()
    if "." not in filename:
        raise ValueError("File must have an extension")

    ext = filename.rsplit(".", 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: .{ext}")

    file_bytes = file_storage.read()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    if ext == "docx":
        return extract_text_from_docx(file_bytes)
    return extract_text_from_txt(file_bytes)


@app.route("/", methods=["GET"])
def index():
    return send_file("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "JSON body must include 'text' field."}), 400

    text = payload.get("text")
    result = analyze_text(text)
    return jsonify({"result": result})


@app.route("/api/analyze-file", methods=["POST"])
def api_analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        text = extract_text_from_file(uploaded_file)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result = analyze_text(text)
    return jsonify({"result": result})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
