"""PDF report generator — APA 7 format with white pages, proper header/footer."""
import os
import csv
import json
import zipfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.units import mm, inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from ..config import RESULTS_DIR


# ── APA 7 Color Palette (professional, grayscale-friendly) ──
APA_BLACK = HexColor("#000000")
APA_DARK_GRAY = HexColor("#333333")
APA_MED_GRAY = HexColor("#666666")
APA_LIGHT_GRAY = HexColor("#e0e0e0")
APA_TABLE_HEADER = HexColor("#4a4a4a")
APA_TABLE_STRIPE = HexColor("#f5f5f5")
APA_WHITE = white
APA_BRAND = HexColor("#0D9488")


class EEGReporter:
    """Generate publishable PDF reports in APA 7 format."""

    def generate(self, study_info: dict, pipeline_outputs: dict, figures: dict,
                 interpretation: str = "", methods: str = "") -> tuple:
        """Generate a full PDF report and ZIP archive. Returns (zip_path, pdf_path, csv_path)."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(RESULTS_DIR, f"cortexiq_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                topMargin=25*mm, bottomMargin=25*mm,
                                leftMargin=25*mm, rightMargin=25*mm)
        styles = getSampleStyleSheet()

        # ── APA 7 Custom Styles ──
        title_style = ParagraphStyle("APATitle", parent=styles["Title"],
                                     textColor=APA_BLACK, fontSize=16,
                                     fontName="Times-Bold", alignment=TA_CENTER,
                                     spaceAfter=6*mm, spaceBefore=2*mm)
        heading1_style = ParagraphStyle("APAHeading1", parent=styles["Heading1"],
                                        textColor=APA_BLACK, fontSize=14,
                                        fontName="Times-Bold", alignment=TA_CENTER,
                                        spaceAfter=4*mm, spaceBefore=6*mm)
        heading2_style = ParagraphStyle("APAHeading2", parent=styles["Heading2"],
                                        textColor=APA_BLACK, fontSize=12,
                                        fontName="Times-Bold", alignment=TA_LEFT,
                                        spaceAfter=3*mm, spaceBefore=5*mm)
        heading3_style = ParagraphStyle("APAHeading3", parent=styles["Heading3"],
                                        textColor=APA_BLACK, fontSize=11,
                                        fontName="Times-BoldItalic", alignment=TA_LEFT,
                                        spaceAfter=2*mm, spaceBefore=4*mm)
        body_style = ParagraphStyle("APABody", parent=styles["Normal"],
                                     fontSize=11, leading=15,
                                     fontName="Times-Roman", alignment=TA_JUSTIFY,
                                     spaceAfter=3*mm)
        note_style = ParagraphStyle("APANote", parent=styles["Normal"],
                                     fontSize=9, leading=12,
                                     fontName="Times-Italic", textColor=APA_MED_GRAY,
                                     spaceAfter=2*mm, alignment=TA_LEFT)
        caption_style = ParagraphStyle("APACaption", parent=styles["Normal"],
                                        fontSize=10, leading=13,
                                        fontName="Times-Italic", textColor=APA_DARK_GRAY,
                                        alignment=TA_LEFT, spaceAfter=3*mm)

        story = []
        stats = pipeline_outputs.get("statistics", {})

        # ── Title Page ──
        story.append(Spacer(1, 30*mm))
        story.append(Paragraph("EEG Analysis Report", title_style))
        study_name = study_info.get("name", "Untitled Study")
        story.append(Paragraph(f"Study: {study_name}", ParagraphStyle("Subtitle",
                               parent=body_style, fontSize=13, fontName="Times-Italic",
                               alignment=TA_CENTER, textColor=APA_DARK_GRAY)))
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            ParagraphStyle("DateLine", parent=body_style, fontSize=11,
                           fontName="Times-Roman", alignment=TA_CENTER, textColor=APA_MED_GRAY)))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            f"Modality: {study_info.get('modality', 'EEG')} | "
            f"Channels: {study_info.get('n_channels', 'N/A')} | "
            f"Sampling Rate: {study_info.get('sfreq', 'N/A')} Hz",
            ParagraphStyle("MetaLine", parent=body_style, fontSize=10,
                           fontName="Times-Roman", alignment=TA_CENTER, textColor=APA_MED_GRAY)))
        story.append(Spacer(1, 15*mm))
        story.append(Paragraph(
            "<i>Report generated by NeuraGentLab's CortexIQ — Neural Signal Analysis Platform</i>",
            ParagraphStyle("BrandLine", parent=body_style, fontSize=10,
                           fontName="Times-Italic", alignment=TA_CENTER, textColor=APA_BRAND)))
        story.append(PageBreak())

        # ── Table helper with APA 7 styling (top-bottom horizontal rules only) ──
        def apa_table(data, col_widths, has_header=True, table_num=None, table_title=None):
            """Create an APA 7 formatted table with proper borders."""
            elements = []
            if table_num and table_title:
                elements.append(Paragraph(f"<b>Table {table_num}</b>", ParagraphStyle(
                    "TableNum", parent=body_style, fontSize=10, fontName="Times-Bold",
                    textColor=APA_BLACK, spaceAfter=1*mm, spaceBefore=4*mm)))
                elements.append(Paragraph(f"<i>{table_title}</i>", ParagraphStyle(
                    "TableTitle", parent=body_style, fontSize=10, fontName="Times-Italic",
                    textColor=APA_BLACK, spaceAfter=2*mm)))

            t = Table(data, colWidths=col_widths, repeatRows=1 if has_header else 0)

            style_commands = [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
                ("TEXTCOLOR", (0, 0), (-1, -1), APA_BLACK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                # APA borders: top line above header, line below header, bottom line
                ("LINEABOVE", (0, 0), (-1, 0), 1.5, APA_BLACK),
                ("LINEBELOW", (0, -1), (-1, -1), 1.5, APA_BLACK),
            ]

            if has_header:
                style_commands.extend([
                    ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.75, APA_BLACK),
                ])

            t.setStyle(TableStyle(style_commands))
            elements.append(t)
            return elements

        # ── 1. Study Parameters ──
        table_num = 1
        story.append(Paragraph("Study Parameters", heading2_style))
        desc_global = stats.get("descriptive", {}).get("global", {})
        study_data = [
            ["Parameter", "Value"],
            ["Study Name", study_name],
            ["Modality", study_info.get("modality", "EEG")],
            ["Number of Channels (N)", str(study_info.get("n_channels", "N/A"))],
            ["Sampling Rate (Hz)", f"{study_info.get('sfreq', 'N/A')}"],
            ["Recording Duration (s)", f"{study_info.get('duration_sec', 'N/A'):.2f}" if isinstance(study_info.get('duration_sec'), (int, float)) else str(study_info.get('duration_sec', 'N/A'))],
            ["Number of Subjects", str(study_info.get("subject_count", 1))],
            ["Total Samples", f"{desc_global.get('n_samples', 'N/A'):,}" if isinstance(desc_global.get('n_samples'), int) else str(desc_global.get('n_samples', 'N/A'))],
            ["Experimental Conditions", study_info.get("conditions", "N/A") or "N/A"],
        ]
        story.extend(apa_table(study_data, [55*mm, 105*mm], table_num=table_num, table_title="Study Configuration and Recording Parameters"))
        story.append(Spacer(1, 4*mm))

        # ── 2. Descriptive Statistics (Global) ──
        if desc_global:
            table_num += 1
            story.append(Paragraph("Descriptive Statistics", heading2_style))
            desc_table = [
                ["Statistic", "Value"],
                ["M (µV)", f"{desc_global.get('global_mean_uV', 0):.4f}"],
                ["SD (µV)", f"{desc_global.get('global_std_uV', 0):.4f}"],
                ["Min (µV)", f"{desc_global.get('global_min_uV', 0):.4f}"],
                ["Max (µV)", f"{desc_global.get('global_max_uV', 0):.4f}"],
                ["Mdn (µV)", f"{desc_global.get('global_median_uV', 0):.4f}"],
                ["σ² (µV²)", f"{desc_global.get('total_variance_uV2', 0):.4f}"],
            ]
            story.extend(apa_table(desc_table, [55*mm, 105*mm], table_num=table_num, table_title="Global Signal Descriptive Statistics"))
            story.append(Paragraph("<i>Note.</i> M = mean; SD = standard deviation; Mdn = median; σ² = variance. Values in microvolts (µV).", note_style))
            story.append(Spacer(1, 4*mm))

        # ── 3. Per-Channel Descriptive Statistics ──
        ch_desc = stats.get("descriptive", {}).get("channels", [])
        if ch_desc:
            table_num += 1
            story.append(Paragraph("Per-Channel Descriptive Statistics", heading2_style))
            ch_header = ["Ch", "M (µV)", "SD (µV)", "Min (µV)", "Max (µV)", "Mdn (µV)", "Skew", "Kurt"]
            ch_rows = [ch_header]
            for ch in ch_desc:
                ch_rows.append([
                    ch["channel"],
                    f"{ch['mean_uV']:.2f}",
                    f"{ch['std_uV']:.2f}",
                    f"{ch['min_uV']:.2f}",
                    f"{ch['max_uV']:.2f}",
                    f"{ch['median_uV']:.2f}",
                    f"{ch['skewness']:.3f}",
                    f"{ch['kurtosis']:.3f}",
                ])
            story.extend(apa_table(ch_rows, [20*mm, 20*mm, 20*mm, 20*mm, 20*mm, 20*mm, 20*mm, 20*mm],
                                   table_num=table_num, table_title="Per-Channel Amplitude Descriptive Statistics"))
            story.append(Paragraph("<i>Note.</i> Ch = channel label; M = mean; SD = standard deviation; Mdn = median; Skew = skewness; Kurt = excess kurtosis. All amplitude values in µV.", note_style))
            story.append(Spacer(1, 4*mm))

        # ── 4. Cross-Channel Correlation ──
        corr = stats.get("descriptive", {}).get("cross_channel_correlation", {})
        if corr and corr.get("mean_r") is not None:
            table_num += 1
            story.append(Paragraph("Cross-Channel Correlation", heading2_style))
            corr_data = [
                ["Statistic", "Value"],
                ["Mean Pearson r", f"{corr['mean_r']:.4f}"],
                ["Min Pearson r", f"{corr.get('min_r', 'N/A'):.4f}" if isinstance(corr.get('min_r'), (int, float)) else "N/A"],
                ["Max Pearson r", f"{corr.get('max_r', 'N/A'):.4f}" if isinstance(corr.get('max_r'), (int, float)) else "N/A"],
                ["Number of Channels", str(len(corr.get("channels", [])))],
            ]
            story.extend(apa_table(corr_data, [55*mm, 105*mm], table_num=table_num, table_title="Cross-Channel Pearson Correlation Summary"))
            story.append(Paragraph("<i>Note.</i> Correlation values computed on the upper triangle of the pairwise Pearson r matrix across all EEG channels.", note_style))
            story.append(Spacer(1, 4*mm))

        # ── 5. Band Power Analysis ──
        band_analysis = stats.get("band_analysis", {})
        if band_analysis:
            table_num += 1
            story.append(Paragraph("Spectral Band Power Analysis", heading2_style))
            bp_header = ["Band", "Frequency (Hz)", "Absolute Power (V²/Hz)", "Relative Power (%)", "Power (dB)"]
            bp_rows = [bp_header]
            band_ranges = {"delta": "1–4", "theta": "4–8", "alpha": "8–13", "beta": "13–30", "gamma": "30–45"}
            for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                if band in band_analysis:
                    b = band_analysis[band]
                    bp_rows.append([
                        band.capitalize(),
                        band_ranges.get(band, ""),
                        f"{b['power_V2_Hz']:.4e}",
                        f"{b['relative_power_pct']:.2f}",
                        f"{b.get('power_dB', 'N/A')}" if b.get('power_dB') is not None else "N/A",
                    ])
            if "total_power_V2_Hz" in band_analysis:
                bp_rows.append(["Total", "1–45", f"{band_analysis['total_power_V2_Hz']:.4e}", "100.00", "—"])
            story.extend(apa_table(bp_rows, [25*mm, 25*mm, 35*mm, 35*mm, 25*mm],
                                   table_num=table_num, table_title="Frequency Band Power Distribution"))
            story.append(Paragraph("<i>Note.</i> Power spectral density computed using Welch's method. Relative power expressed as percentage of total spectral power across all bands.", note_style))
            story.append(Spacer(1, 4*mm))

        # ── 6. Per-Channel Band Powers ──
        per_ch = band_analysis.get("per_channel", []) if band_analysis else []
        if per_ch:
            table_num += 1
            story.append(Paragraph("Per-Channel Band Power Distribution", heading2_style))
            bands_list = ["delta", "theta", "alpha", "beta", "gamma"]
            pch_header = ["Channel"] + [f"{b.capitalize()} (V²/Hz)" for b in bands_list]
            pch_rows = [pch_header]
            for row in per_ch:
                pch_rows.append(
                    [row["channel"]] + [f"{row.get(f'{b}_power', 0):.2e}" for b in bands_list]
                )
            col_w = [25*mm] + [27*mm] * len(bands_list)
            story.extend(apa_table(pch_rows, col_w, table_num=table_num,
                                   table_title="Per-Channel Absolute Band Power"))
            story.append(Paragraph("<i>Note.</i> Absolute power values in V²/Hz for each standard EEG frequency band per channel.", note_style))
            story.append(Spacer(1, 4*mm))

        # ── 7. ERP Analysis ──
        erp_analysis = stats.get("erp_analysis", {})
        if erp_analysis:
            table_num += 1
            story.append(Paragraph("Event-Related Potential Analysis", heading2_style))
            erp_table = [
                ["Parameter", "Value"],
                ["Peak Channel", erp_analysis.get("peak_channel", "N/A")],
                ["Peak Latency (ms)", f"{erp_analysis.get('peak_latency_ms', 0):.2f}"],
                ["Peak Amplitude (µV)", f"{erp_analysis.get('peak_amplitude_uV', 0):.2f}"],
            ]
            story.extend(apa_table(erp_table, [55*mm, 105*mm], table_num=table_num,
                                   table_title="ERP Peak Detection Results"))
            story.append(Paragraph("<i>Note.</i> Peak determined by maximum absolute amplitude across all channels and time points.", note_style))
            story.append(Spacer(1, 4*mm))

        # ── 8. Epoch Analysis ──
        epoch_analysis = stats.get("epoch_analysis", {})
        if epoch_analysis:
            table_num += 1
            story.append(Paragraph("Epoching Summary", heading2_style))
            ep_table = [
                ["Parameter", "Value"],
                ["Number of Epochs", str(epoch_analysis.get("n_epochs", "N/A"))],
                ["Epoch Duration (s)", f"{epoch_analysis.get('epoch_duration_sec', 0):.3f}"],
                ["Time Window (s)", f"{epoch_analysis.get('tmin_sec', 0):.3f} to {epoch_analysis.get('tmax_sec', 0):.3f}"],
                ["Total Epoch Time (s)", f"{epoch_analysis.get('total_epoch_time_sec', 0):.3f}"],
            ]
            story.extend(apa_table(ep_table, [55*mm, 105*mm], table_num=table_num,
                                   table_title="Epoch Segmentation Parameters"))
            story.append(Spacer(1, 4*mm))

        # ── 9. Pipeline Steps ──
        steps = pipeline_outputs.get("steps", {})
        if steps:
            table_num += 1
            story.append(Paragraph("Processing Pipeline", heading2_style))
            step_rows = [["Step", "Procedure", "Status", "Description"]]
            for step_id, step_data in steps.items():
                if isinstance(step_data, dict):
                    name = step_data.get("name", f"Step {step_id}")
                    summary = step_data.get("summary", "")
                    status = step_data.get("status", "")
                    step_rows.append([str(int(step_id) + 1), name, status.capitalize(), summary[:80]])
            story.extend(apa_table(step_rows, [12*mm, 45*mm, 20*mm, 83*mm],
                                   table_num=table_num, table_title="Signal Processing Pipeline Steps"))
            story.append(Spacer(1, 4*mm))

        # ── Figures ──
        fig_count = 0
        figure_labels = {
            "psd": "Power Spectral Density",
            "erp": "Event-Related Potential (Butterfly Plot)",
            "ica": "Independent Component Analysis",
            "topomap": "Topographic Band Power Distribution"
        }
        for fig_name, fig_path in figures.items():
            if os.path.exists(fig_path):
                fig_count += 1
                story.append(PageBreak())
                story.append(Paragraph(f"<b>Figure {fig_count}</b>", ParagraphStyle(
                    "FigureNum", parent=body_style, fontSize=10, fontName="Times-Bold",
                    textColor=APA_BLACK, spaceAfter=1*mm, spaceBefore=4*mm)))
                fig_label = figure_labels.get(fig_name, fig_name.replace("_", " ").title())
                story.append(Paragraph(f"<i>{fig_label}</i>", ParagraphStyle(
                    "FigureTitle", parent=body_style, fontSize=10, fontName="Times-Italic",
                    textColor=APA_BLACK, spaceAfter=3*mm)))
                try:
                    img = Image(fig_path, width=155*mm, height=85*mm)
                    img.hAlign = 'CENTER'
                    story.append(img)
                    story.append(Spacer(1, 3*mm))
                    story.append(Paragraph(
                        f"<i>Note.</i> {fig_label} generated during signal processing pipeline. "
                        f"All figures produced using MNE-Python and Matplotlib.",
                        note_style))
                except Exception:
                    story.append(Paragraph(f"(Figure {fig_name} could not be embedded)", body_style))

        # ── Interpretation ──
        if interpretation:
            story.append(PageBreak())
            story.append(Paragraph("Results Interpretation", heading2_style))
            # Split interpretation into paragraphs for readability
            for para in interpretation.split(". "):
                clean = para.strip()
                if clean:
                    if not clean.endswith("."):
                        clean += "."
                    story.append(Paragraph(clean, body_style))

        # ── Methods ──
        if methods:
            story.append(Spacer(1, 6*mm))
            story.append(Paragraph("Method", heading2_style))
            story.append(Paragraph(methods, body_style))

        # Build PDF with header/footer
        def _header_footer(canvas, doc):
            """Draw running header and footer on each page."""
            canvas.saveState()
            page_w, page_h = A4

            # Header: "NeuraGentLab's CortexIQ" on left, page number on right
            canvas.setFont("Times-Roman", 9)
            canvas.setFillColor(APA_MED_GRAY)
            canvas.drawString(25*mm, page_h - 15*mm, "NeuraGentLab's CortexIQ")
            canvas.drawRightString(page_w - 25*mm, page_h - 15*mm, f"{doc.page}")
            # Header line
            canvas.setStrokeColor(APA_LIGHT_GRAY)
            canvas.setLineWidth(0.5)
            canvas.line(25*mm, page_h - 17*mm, page_w - 25*mm, page_h - 17*mm)

            # Footer
            canvas.setFont("Times-Roman", 8)
            canvas.setFillColor(APA_MED_GRAY)
            canvas.drawCentredString(page_w / 2, 15*mm,
                                     "NeuraGentLab's CortexIQ — Neural Signal Analysis Platform")
            # Footer line
            canvas.line(25*mm, 18*mm, page_w - 25*mm, 18*mm)

            canvas.restoreState()

        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

        # Save comprehensive CSV
        csv_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
        self._save_csv(csv_path, stats, pipeline_outputs, study_info)

        # Save JSON
        json_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
        json_data = {"study_info": study_info, "statistics": stats,
                     "band_powers": pipeline_outputs.get("band_powers", {}),
                     "erp_peak": pipeline_outputs.get("erp_peak", {})}
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        # Create ZIP
        zip_path = os.path.join(RESULTS_DIR, f"cortexiq_results_{timestamp}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pdf_path, os.path.basename(pdf_path))
            zf.write(csv_path, os.path.basename(csv_path))
            zf.write(json_path, os.path.basename(json_path))
            for fig_name, fig_path in figures.items():
                if os.path.exists(fig_path):
                    zf.write(fig_path, os.path.basename(fig_path))

        return zip_path, pdf_path, csv_path

    def _save_csv(self, path: str, stats: dict, outputs: dict, study_info: dict = None):
        """Save comprehensive numerical results to CSV with APA-style labeled sections."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header row
            writer.writerow(["NeuraGentLab's CortexIQ - EEG Analysis Results"])
            writer.writerow([f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            if study_info:
                writer.writerow([f"Study: {study_info.get('name', 'N/A')}"])
            writer.writerow([])

            # Section 1: Global Descriptive Statistics
            writer.writerow(["Table 1: Global Descriptive Statistics"])
            writer.writerow(["Statistic", "Value", "Unit"])
            desc_global = stats.get("descriptive", {}).get("global", {})
            stat_labels = {
                "n_channels": ("Number of Channels (N)", "", ""),
                "n_samples": ("Total Samples", "", ""),
                "duration_sec": ("Recording Duration", "", "s"),
                "sampling_rate_Hz": ("Sampling Rate", "", "Hz"),
                "global_mean_uV": ("M (Global Mean)", "", "µV"),
                "global_std_uV": ("SD (Global Std Dev)", "", "µV"),
                "global_min_uV": ("Min (Global Minimum)", "", "µV"),
                "global_max_uV": ("Max (Global Maximum)", "", "µV"),
                "global_median_uV": ("Mdn (Global Median)", "", "µV"),
                "total_variance_uV2": ("σ² (Total Variance)", "", "µV²"),
            }
            for key, val in desc_global.items():
                label_info = stat_labels.get(key, (key.replace("_", " ").title(), "", ""))
                writer.writerow([label_info[0], val, label_info[2]])
            writer.writerow([])

            # Section 2: Per-Channel Descriptive Statistics
            ch_desc = stats.get("descriptive", {}).get("channels", [])
            if ch_desc:
                writer.writerow(["Table 2: Per-Channel Descriptive Statistics"])
                writer.writerow(["Index", "Channel", "M (µV)", "SD (µV)", "Min (µV)", "Max (µV)",
                                 "Mdn (µV)", "Q1 (µV)", "Q3 (µV)", "IQR (µV)",
                                 "Skewness", "Kurtosis", "RMS (µV)", "σ² (µV²)"])
                for ch in ch_desc:
                    writer.writerow([
                        ch.get("index", ""), ch["channel"], ch["mean_uV"], ch["std_uV"],
                        ch["min_uV"], ch["max_uV"], ch["median_uV"], ch["q25_uV"],
                        ch["q75_uV"], ch["iqr_uV"], ch["skewness"], ch["kurtosis"],
                        ch["rms_uV"], ch["variance_uV2"],
                    ])
                writer.writerow([])

            # Section 3: Cross-Channel Correlation Matrix
            corr = stats.get("descriptive", {}).get("cross_channel_correlation", {})
            if corr and corr.get("matrix"):
                writer.writerow(["Table 3: Cross-Channel Pearson Correlation Matrix"])
                writer.writerow([""] + corr["channels"])
                for i, row in enumerate(corr["matrix"]):
                    writer.writerow([corr["channels"][i]] + row)
                writer.writerow([])
                writer.writerow(["Summary", ""])
                writer.writerow(["Mean r", corr.get("mean_r", "")])
                writer.writerow(["Min r", corr.get("min_r", "")])
                writer.writerow(["Max r", corr.get("max_r", "")])
                writer.writerow([])

            # Section 4: Band Power Analysis
            band_analysis = stats.get("band_analysis", {})
            if band_analysis:
                writer.writerow(["Table 4: Spectral Band Power Analysis"])
                writer.writerow(["Band", "Frequency Range (Hz)", "Absolute Power (V²/Hz)",
                                "Relative Power (%)", "Power (dB)"])
                band_ranges = {"delta": "1–4", "theta": "4–8", "alpha": "8–13", "beta": "13–30", "gamma": "30–45"}
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    if band in band_analysis:
                        b = band_analysis[band]
                        writer.writerow([band.capitalize(), band_ranges.get(band, ""),
                                        b["power_V2_Hz"], b["relative_power_pct"],
                                        b.get("power_dB", "")])
                if "total_power_V2_Hz" in band_analysis:
                    writer.writerow(["Total", "1–45", band_analysis["total_power_V2_Hz"], "100.00", ""])
                writer.writerow([])

                # Per-channel band powers
                per_ch = band_analysis.get("per_channel", [])
                if per_ch:
                    writer.writerow(["Table 5: Per-Channel Band Powers (V²/Hz)"])
                    bands = ["delta", "theta", "alpha", "beta", "gamma"]
                    writer.writerow(["Index", "Channel"] + [f"{b.capitalize()} (V²/Hz)" for b in bands])
                    for row in per_ch:
                        writer.writerow([row.get("index", ""), row["channel"]] +
                                       [row.get(f"{b}_power", "") for b in bands])
                    writer.writerow([])

            # Section 5: ERP Analysis
            erp_analysis = stats.get("erp_analysis", {})
            if erp_analysis:
                writer.writerow(["Table 6: Event-Related Potential Analysis"])
                writer.writerow(["Parameter", "Value", "Unit"])
                erp_labels = {
                    "peak_channel": ("Peak Channel", ""),
                    "peak_latency_ms": ("Peak Latency", "ms"),
                    "peak_amplitude_uV": ("Peak Amplitude", "µV"),
                }
                for key, val in erp_analysis.items():
                    label_info = erp_labels.get(key, (key.replace("_", " ").title(), ""))
                    writer.writerow([label_info[0], val, label_info[1]])
                writer.writerow([])

            # Section 6: Epoch Analysis
            epoch_analysis = stats.get("epoch_analysis", {})
            if epoch_analysis:
                writer.writerow(["Table 7: Epoch Analysis"])
                writer.writerow(["Parameter", "Value", "Unit"])
                epoch_labels = {
                    "n_epochs": ("Number of Epochs", ""),
                    "epoch_duration_sec": ("Epoch Duration", "s"),
                    "tmin_sec": ("Epoch Start Time", "s"),
                    "tmax_sec": ("Epoch End Time", "s"),
                    "total_epoch_time_sec": ("Total Epoch Time", "s"),
                }
                for key, val in epoch_analysis.items():
                    label_info = epoch_labels.get(key, (key.replace("_", " ").title(), ""))
                    writer.writerow([label_info[0], val, label_info[1]])
                writer.writerow([])

            # Section 7: Pipeline Step Results
            steps = outputs.get("steps", {})
            if steps:
                writer.writerow(["Table 8: Processing Pipeline Steps"])
                writer.writerow(["Step", "Procedure", "Status", "Description"])
                for sid, sdata in steps.items():
                    if isinstance(sdata, dict):
                        writer.writerow([int(sid) + 1, sdata.get("name", ""), sdata.get("status", ""),
                                        sdata.get("summary", "")])
