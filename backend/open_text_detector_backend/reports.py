import csv
import io
import json
from xml.sax.saxutils import escape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def safe_csv(value):
    value = str(value or "")
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def csv_report(scans):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "title", "status", "model", "score", "label", "words", "created_at"])
    for scan in scans:
        result = scan.get("result") or {}
        writer.writerow([safe_csv(v) for v in [scan["id"], scan["title"], scan["status"], scan["model"]["name"],
            result.get("score"), result.get("label"), scan["word_count"], scan["created_at"]]])
    return stream.getvalue()


def pdf_report(scan):
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#214d3b")
    styles["BodyText"].leading = 15
    result = scan.get("result") or {}
    blocks = [Paragraph("Open Text Detector Workbench · Scan report", styles["Title"]), Spacer(1, 14),
              Paragraph(escape(scan["title"]), styles["Heading2"])]
    for label, value in [("Model", scan["model"]["name"]), ("Status", scan["status"]),
                         ("Created", scan["created_at"]), ("Words", scan["word_count"]),
                         ("Score", result.get("score", "Not available")), ("Classification", result.get("label", "Not available"))]:
        blocks.append(Paragraph(f"<b>{label}:</b> {escape(str(value))}", styles["BodyText"]))
    blocks.append(Paragraph(escape(result.get("notice", "No classification result is available.")), styles["BodyText"]))
    if result.get("plagiarism"):
        blocks.append(Paragraph(escape(result["plagiarism"]["notice"]), styles["BodyText"]))
    blocks.append(Spacer(1, 18))
    for paragraph in scan["text"].split("\n"):
        # Bound individual Paragraph objects for extremely long unbroken inputs.
        for start in range(0, len(paragraph), 3000):
            blocks.append(Paragraph(escape(paragraph[start:start+3000]), styles["BodyText"]))
    SimpleDocTemplate(output, title=scan["title"], author="Open Text Detector Workbench").build(blocks)
    return output.getvalue()


def json_report(scan):
    return json.dumps(scan, indent=2, ensure_ascii=False)
