"""Render a six-page report from measured results and a reviewed visual note."""
import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage1.runtime import ROOT as PROJECT_ROOT

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def register_fonts():
    candidates = [
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Report", str(regular)))
            pdfmetrics.registerFont(TTFont("Report-Bold", str(bold)))
            pdfmetrics.registerFontFamily("Report", normal="Report", bold="Report-Bold", italic="Report", boldItalic="Report-Bold")
            return
    raise RuntimeError("A Cyrillic font is required: DejaVu Sans (Linux) or Arial (macOS)")


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font size="8.8">\1</font>', text)
    return text


def build_pdf(markdown, destination):
    register_fonts()
    base = dict(fontName="Report", fontSize=9.6, leading=12.8, spaceAfter=6, splitLongWords=True)
    styles = {
        "body": ParagraphStyle("Body", **base),
        "h1": ParagraphStyle("Title", fontName="Report-Bold", fontSize=17, leading=21, spaceAfter=12),
        "h2": ParagraphStyle("Heading", fontName="Report-Bold", fontSize=12, leading=15, spaceBefore=7, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("Subheading", fontName="Report-Bold", fontSize=10.2, leading=13, spaceBefore=5, spaceAfter=5, keepWithNext=True),
        "table": ParagraphStyle("Cell", fontName="Report", fontSize=8.5, leading=10.8, spaceAfter=0),
        "caption": ParagraphStyle("Caption", fontName="Report", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=6),
    }
    story = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["h1"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["h3"]))
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            path = ROOT / "report" / match.group(2)
            if not path.exists():
                raise FileNotFoundError(f"Required report image is absent: {path}")
            story.append(Image(str(path), width=220, height=220))
            story.append(Paragraph(inline(match.group(1)), styles["caption"]))
        elif line.startswith("|"):
            rows = [line]
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            cells = [[Paragraph(inline(cell.strip()), styles["table"]) for cell in row.strip("|").split("|")] for row in rows if not re.fullmatch(r"[| :\-]+", row)]
            widths = [225, 280] if len(cells[0]) == 2 else [140, 190, 175]
            table = Table(cells, colWidths=widths, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff5")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c5d0da")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d6dee5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 7)])
        else:
            if line.startswith("- "):
                line = "• " + line[2:]
            story.append(Paragraph(inline(line), styles["body"]))

    def footer(canvas, document):
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#506070"))
        canvas.drawString(45, 27, "Разные методы генерации из диффузионных моделей · Этап 1")
        canvas.drawRightString(A4[0] - 45, 27, str(document.page))

    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=36, bottomMargin=43,
                                 title="Разные методы генерации из диффузионных моделей — этап 1", author="")
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/baseline/results.json")
    args = parser.parse_args()
    results_path = ROOT / args.results
    result = json.loads(results_path.read_text())
    if result.get("status") != "complete":
        raise RuntimeError("A complete measured baseline with FID is required before report generation")
    config, generation, fid = result["config"], result["generation"], result["fid"]
    hardware_path = results_path.parent / "hardware.json"
    hardware = json.loads(hardware_path.read_text())["description"] if hardware_path.exists() else result["environment"]["accelerator"]
    benchmark = json.loads((results_path.parent / "benchmark.json").read_text()) if (results_path.parent / "benchmark.json").exists() else None
    verification_path = results_path.parent / "fid_verification.json"
    verification_note = ""
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        if not verification["passed"]:
            raise RuntimeError("Independent FID verification did not pass")
        verification_note = f'FID дополнительно проверен независимым расчётом через собственные значения симметричной положительно полуопределённой матрицы; абсолютное расхождение с torch-fidelity составило {verification["absolute_difference"]:.2g}.'
    note_path = ROOT / "report" / "visual_analysis.txt"
    if not note_path.exists():
        raise RuntimeError("The generated grid must be reviewed and described in report/visual_analysis.txt")
    values = {
        "DEVICE": result["environment"]["device"], "PARAMS": "35,7 млн",
        "HARDWARE": hardware,
        "PYTHON_VERSION": result["environment"]["python"],
        "TORCH_VERSION": result["environment"]["versions"]["torch"],
        "DIFFUSERS_VERSION": result["environment"]["versions"]["diffusers"],
        "NUM_REAL": fid["num_real"], "NUM_GENERATED": fid["num_generated"],
        "NOISE_SHA": result["noise_sha256"], "STEPS": config["steps"],
        "NFE": generation["nfe_per_image"], "BATCH_CALLS": generation["actual_batched_unet_calls"],
        "GEN_SECONDS": f'{generation["total_seconds"]:.2f}', "GEN_MINUTES": f'{generation["total_seconds"]/60:.2f}',
        "SECONDS_IMAGE": f'{generation["seconds_per_image"]:.4f}', "FID": f'{fid["frechet_inception_distance"]:.4f}',
        "FID_SECONDS": f'{fid["evaluation_seconds"]:.2f}', "PIPELINE_ERROR": f'{result["pipeline_check"]["max_absolute_error"]:.2g}',
        "BATCH_COUNT": generation["batch_count"], "BATCH_SIZE": config["batch_size"],
        "SAMPLE_EVALUATIONS": generation["sample_model_evaluations"],
        "RESULT_STATUS": "Baseline воспроизведён с готовыми весами и настройками раздела 3. Получены численные метрики и сохранены изображения. Значения таблицы прочитаны из results.json.",
        "FID_VERIFICATION": verification_note,
        "VISUAL_ANALYSIS": note_path.read_text().strip(),
        "BENCHMARK_DESCRIPTION": (f'В коротком benchmark медиана трёх повторов составила {benchmark["median_seconds"]:.3f} с на батч из {benchmark["batch_size"]} изображений и {benchmark["benchmark_steps"]} шагов.' if benchmark else "Отдельный короткий benchmark в этом каталоге отсутствует."),
    }
    markdown = (ROOT / "report" / "template.md").read_text()
    for key, value in values.items():
        markdown = markdown.replace("{{" + key + "}}", str(value))
    if "{{" in markdown:
        raise RuntimeError("Unresolved report placeholders")
    import os
    relative_artifacts = os.path.relpath(results_path.parent, ROOT / "report")
    markdown = markdown.replace("../artifacts/baseline", relative_artifacts)
    (ROOT / "report" / "stage1_report.md").write_text(markdown)
    build_pdf(markdown, ROOT / "report" / "stage1_report.pdf")
    print("Built report/stage1_report.md and report/stage1_report.pdf from measured results")


if __name__ == "__main__":
    main()
