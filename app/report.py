"""Images (slices, nodule close-ups, overview) and the PDF report for doctors."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

RISK_COLORS = {"High": (230, 57, 70), "Intermediate": (244, 162, 97), "Low": (42, 157, 143), "n/a": (90, 160, 255)}


def window(a, lo=-1350, hi=150):
    return (np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _font(size=12):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def take_slice(vol, axis, idx):
    if axis == "axial":
        return vol[idx]
    if axis == "coronal":
        return vol[:, idx, :][::-1]      # superior at top
    return vol[:, :, idx][::-1]           # sagittal


def render_slice(arr, labels, lung, axis, idx, nodules, show_lung=False, scale=2):
    """PNG bytes of one slice with nodule outlines (coloured by AI risk) and ID tags."""
    n = {"axial": arr.shape[0], "coronal": arr.shape[1], "sagittal": arr.shape[2]}[axis]
    idx = int(np.clip(idx, 0, n - 1))
    g = window(take_slice(arr, axis, idx))
    rgb = np.stack([g] * 3, -1)
    lab = take_slice(labels, axis, idx)
    if show_lung:
        lm = take_slice(lung, axis, idx)
        edge = lm & ~ndi.binary_erosion(lm)
        rgb[edge] = (80, 140, 255)
    by_id = {int(r["candidate_id"]): r for r in nodules}
    for cid in np.unique(lab):
        if cid == 0 or int(cid) not in by_id:
            continue
        m = lab == cid
        edge = m & ~ndi.binary_erosion(m)
        edge = ndi.binary_dilation(edge)
        rgb[edge] = RISK_COLORS.get(by_id[int(cid)].get("ai_risk", "n/a"), (90, 160, 255))
    im = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(im)
    f = _font(13)
    for cid in np.unique(lab):
        if cid == 0 or int(cid) not in by_id:
            continue
        ys, xs = np.where(lab == cid)
        x, y = xs.max() * scale + 6, ys.min() * scale - 4
        r = by_id[int(cid)]
        c = RISK_COLORS.get(r.get("ai_risk", "n/a"), (90, 160, 255))
        d.text((x, y), f"#{r['display_id']}", fill=c, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def nodule_closeup(arr, labels, row, half=None, scale=None):
    """Axial | coronal | sagittal close-up of one nodule, PNG bytes."""
    z, y, x = int(row["voxel_z"]), int(row["voxel_y"]), int(row["voxel_x"])
    cid = int(row["candidate_id"])
    if half is None:  # zoom so the nodule fills roughly a third of the tile
        half = int(np.clip(row["diameter_mm"] * 1.6, 9, 40))
    if scale is None:
        scale = max(2, int(round(192 / (2 * half))))
    color = RISK_COLORS.get(row.get("ai_risk", "n/a"), (90, 160, 255))
    tiles = []

    def crop2(a, c0, c1):
        pad = np.pad(a, half, mode="constant", constant_values=a.min() if a.dtype != bool else 0)
        return pad[c0:c0 + 2 * half, c1:c1 + 2 * half]

    for img2, lab2, c0, c1, flip in ((arr[z], labels[z], y, x, False),
                                     (arr[:, y, :], labels[:, y, :], z, x, True),
                                     (arr[:, :, x], labels[:, :, x], z, y, True)):
        a = crop2(img2, c0, c1)
        m = crop2(lab2, c0, c1) == cid
        if flip:
            a, m = a[::-1], m[::-1]
        g = window(a)
        rgb = np.stack([g] * 3, -1)
        e = m & ~ndi.binary_erosion(m)
        rgb[e] = color
        tiles.append(Image.fromarray(rgb).resize((2 * half * scale, 2 * half * scale), Image.NEAREST))
    W = sum(t.width for t in tiles) + 8 * 2
    out = Image.new("RGB", (W, tiles[0].height), (20, 24, 32))
    xo = 0
    f = _font(12)
    for t, name in zip(tiles, ("Axial", "Coronal", "Sagittal")):
        out.paste(t, (xo, 0))
        dr = ImageDraw.Draw(out)
        dr.rectangle([xo, 0, xo + 64, 20], fill=(20, 24, 32))
        dr.text((xo + 6, 4), name, fill=(230, 230, 230), font=f)
        xo += t.width + 8
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def overview_png(ov):
    """Coronal projection of the whole uploaded scan with the extracted lung region boxed."""
    mip = ov["coronal_slice"][::-1]
    g = window(mip, -1100, 400)
    rgb = np.stack([g] * 3, -1)
    lm = ov["coronal_lung"][::-1]
    edge = lm & ~ndi.binary_erosion(lm)
    rgb[edge] = (60, 130, 255)
    im = Image.fromarray(rgb)
    sc = 2
    im = im.resize((im.width * sc, im.height * sc), Image.BILINEAR)
    z0, z1, x0, x1 = ov["bbox_zx"]
    H = mip.shape[0]
    d = ImageDraw.Draw(im)
    d.rectangle([x0 * sc, (H - 1 - z1) * sc, x1 * sc, (H - 1 - z0) * sc], outline=(255, 209, 102), width=3)
    d.text((x0 * sc + 4, (H - 1 - z1) * sc + 4), "Extracted lungs", fill=(255, 209, 102), font=_font(14))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------
def build_pdf(path: Path, result: dict, closeups: dict, overview: bytes | None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, spaceAfter=2, textColor=colors.HexColor("#0b2545"))
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4,
                        textColor=colors.HexColor("#13315c"))
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.5)
    S = ParagraphStyle("s", parent=B, fontSize=7.8, leading=10, textColor=colors.HexColor("#555555"))
    WARN = ParagraphStyle("w", parent=B, textColor=colors.HexColor("#9b2226"))

    def tbl(data, widths, header=True, zebra=True):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("FONTSIZE", (0, 0), (-1, -1), 8.3), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d3df")),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13315c")),
                   ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        if zebra:
            for r in range(1 if header else 0, len(data)):
                if r % 2 == 0:
                    st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f2f5f9")))
        t.setStyle(TableStyle(st))
        return t

    scan, lungs, nods = result["scan"], result["lungs"], result["nodules"]
    story = [Paragraph("LungLocAI &mdash; CT Lung Nodule Analysis Report", H1),
             Paragraph(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; Job {result['job_id']}", S),
             Spacer(1, 4),
             Paragraph("<b>Decision-support output from a research prototype.</b> Findings are computer-generated and "
                       "must be verified by a qualified radiologist on the original images before any clinical use.", WARN)]
    demo_models = [k for k, v in result["models"].get("details", {}).items() if v.get("demo")]
    if demo_models:
        what = " and ".join("cancer probability" if "malign" in k else "nodule detection" for k in demo_models)
        story.append(Paragraph(f"<b>DEMO MODEL:</b> the {what} model was trained on synthetic phantoms, not on patient "
                               "data, so those probabilities are illustrative only.", WARN))

    story.append(Paragraph("1. Scan", H2))
    story.append(tbl([
        ["Source", scan.get("source_file", ""), "Format", scan.get("format", "")],
        ["Series UID", Paragraph(str(scan.get("series_uid", ""))[:80], S), "Voxel spacing (mm)",
         " x ".join(str(s) for s in scan.get("spacing_mm", []))],
        ["Scan coverage", f"{lungs['scan_length_mm']:.0f} mm", "Scan type",
         Paragraph("Full-body / extended CT (lungs auto-extracted)" if lungs["full_body"] else "Chest CT", S)],
        ["Lung volume", f"{lungs['lung_volume_ml']:.0f} mL", "Candidates examined", str(result["n_candidates"])],
    ], [30 * mm, 60 * mm, 35 * mm, 55 * mm], header=False))

    story.append(Paragraph("2. Impression", H2))
    for line in result["impression"]:
        story.append(Paragraph("&bull; " + line, B))
    if overview is not None and lungs["full_body"]:
        img = RLImage(io.BytesIO(overview))
        r = img.imageHeight / img.imageWidth
        h = 62 * mm
        img.drawWidth, img.drawHeight = h / r, h
        story += [Spacer(1, 4), img, Paragraph("Coronal overview of the uploaded scan; yellow box = region extracted "
                                               "as lungs for analysis.", S)]

    story.append(Paragraph("3. Findings", H2))
    if not nods:
        story.append(Paragraph("No lung nodules were reported by the system.", B))
    else:
        has_p = any(n.get("p_malignant") is not None for n in nods)
        rows = [["#", "Location", "Size (mm)", "Type", "Lung-RADS*", "AI cancer prob." if has_p else "AI label",
                 "AI risk", "Priority"]]
        for n in nods:
            p = n.get("p_malignant")
            rows.append([n["display_id"], f"{n['side']} lung, {n['zone'].lower()} zone", f"{n['diameter_mm']:.1f}",
                         n["nodule_type"], n["lungrads"], f"{p * 100:.0f}%" if p is not None else n["ai_label"],
                         n["ai_risk"], n["priority"]])
        t = tbl(rows, [8 * mm, 40 * mm, 17 * mm, 21 * mm, 19 * mm, 24 * mm, 22 * mm, 30 * mm])
        st = []
        for i, n in enumerate(nods, start=1):
            c = {"High": "#e63946", "Intermediate": "#f4a261", "Low": "#2a9d8f"}.get(n["ai_risk"])
            if c:
                st.append(("TEXTCOLOR", (6, i), (6, i), colors.HexColor(c)))
                st.append(("FONTNAME", (6, i), (6, i), "Helvetica-Bold"))
        t.setStyle(TableStyle(st))
        story.append(t)
        story.append(Paragraph("*Automatic size/type category in the style of ACR Lung-RADS v2022 (baseline). "
                               "Size = volume-equivalent diameter from 3-D segmentation.", S))

        story.append(PageBreak())
        story.append(Paragraph("4. Nodule details", H2))
        for n in nods:
            block = [Paragraph(f"<b>Nodule #{n['display_id']}</b> &mdash; {n['side']} lung, {n['zone'].lower()} zone "
                               f"&mdash; {n['ai_label']}", B)]
            if n["candidate_id"] in closeups:
                im = RLImage(io.BytesIO(closeups[n["candidate_id"]]))
                im.drawWidth, im.drawHeight = 150 * mm, 150 * mm * im.imageHeight / im.imageWidth
                block.append(im)
            p = n.get("p_malignant")
            block.append(tbl([
                ["Diameter", f"{n['diameter_mm']:.1f} mm", "Volume", Paragraph(f"{n['volume_mm3']:.0f} mm<super>3</super>", B),
                 "Long / short axis", f"{n['major_axis_mm']:.1f} / {n['minor_axis_mm']:.1f} mm"],
                ["Mean density", f"{n['mean_hu']:.0f} HU", "Solid fraction", f"{n['solid_fraction'] * 100:.0f}%",
                 "Calcified fraction", f"{n['calcified_fraction'] * 100:.0f}%"],
                ["Sphericity", f"{n['sphericity']:.2f}", "Margin irregularity", f"{n['margin_irregularity']:.2f}",
                 "Pleura distance", f"{n['pleura_dist_mm']:.1f} mm" + (" (juxtapleural)" if n["juxtapleural"] else "")],
                ["World coord (mm)", f"{n['world_x']:.1f}, {n['world_y']:.1f}, {n['world_z']:.1f}",
                 "AI cancer prob.", f"{p * 100:.1f}%" if p is not None else "n/a",
                 "Nodule prob.", f"{n['p_nodule'] * 100:.0f}%" if n.get("p_nodule") is not None else "n/a"],
            ], [27 * mm, 33 * mm, 27 * mm, 27 * mm, 29 * mm, 37 * mm], header=False))
            block.append(Paragraph(f"<b>Suggested management (Lung-RADS {n['lungrads']}-style):</b> {n['management']}", B))
            block.append(Spacer(1, 8))
            story.append(KeepTogether(block))

    story.append(Paragraph("5. Method &amp; model", H2))
    m = result["models"]
    story.append(Paragraph(
        "CT converted to a 1 mm isotropic 3-D volume &rarr; lungs segmented (and automatically extracted from "
        "full-body scans) &rarr; nodule candidates found with multi-scale 3-D morphology &rarr; each candidate "
        "described by the 539 LungLocAI-3D features (32 mm cube at 1 mm: sphere statistics, contrast, gradient, "
        f"8&times;8&times;8 patch) plus {len(result['feature_columns'])} shape/density/location measurements &rarr; XGBoost "
        "classification (nodule model: LungLocAI-3D, IEEE DataPort DOI 10.21227/0x9w-1y93).", B))
    for name, info in m.get("details", {}).items():
        met = info.get("metrics") or {}
        story.append(Paragraph(
            f"<b>{name}</b> ({info.get('task')}): trained on {info.get('trained_on')} "
            f"({info.get('n_rows')} rows, {info.get('n_scans')} scans){' — DEMO' if info.get('demo') else ''}. "
            f"Cross-validated AUC {met.get('auc', 'n/a')}, sensitivity {met.get('sensitivity', 'n/a')}, "
            f"specificity {met.get('specificity', 'n/a')}; decision threshold {info.get('threshold')}.", B))
    for w in result.get("warnings", []):
        story.append(Paragraph("Note: " + w, S))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Limitations: nodules fused with the mediastinum/heart or diaphragm can be missed; very "
                           "faint ground-glass nodules (&lt; -600 HU) may be under-segmented; the model is only as good "
                           "as its training data.", S))
    story.append(Spacer(1, 18))
    story.append(tbl([["Reviewed by (radiologist):", "", "Date:", ""]], [45 * mm, 70 * mm, 15 * mm, 50 * mm],
                     header=False, zebra=False))

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(15 * mm, 9 * mm, "LungLocAI research prototype — not a medical device. Verify all findings.")
        c.drawRightString(195 * mm, 9 * mm, f"Page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=15 * mm, title="LungLocAI report")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
