"""PDF report generator with embedded figures and AI interpretation."""
import os
import csv
import json
import zipfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from ..config import RESULTS_DIR


class EEGReporter:
    """Generate PDF reports with figures, metrics, and AI interpretation."""

    def generate(self, study_info: dict, pipeline_outputs: dict, figures: dict,
                 interpretation: str = "", methods: str = "") -> tuple:
        """Generate a full PDF report and ZIP archive. Returns (zip_path, pdf_path, csv_path)."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(RESULTS_DIR, f"cortexiq_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                topMargin=20*mm, bottomMargin=20*mm,
                                leftMargin=15*mm, rightMargin=15*mm)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle("CortexTitle", parent=styles["Title"],
                                     textColor=HexColor("#0D9488"), fontSize=20)
        heading_style = ParagraphStyle("CortexHeading", parent=styles["Heading2"],
                                        textColor=HexColor("#10B981"), fontSize=13)
        subheading_style = ParagraphStyle("CortexSubHeading", parent=styles["Heading3"],
                                           textColor=HexColor("#0D9488"), fontSize=11)
        body_style = ParagraphStyle("CortexBody", parent=styles["Normal"],
                                     fontSize=9, leading=12)
        small_style = ParagraphStyle("CortexSmall", parent=styles["Normal"],
                                      fontSize=8, leading=10, textColor=HexColor("#64748b"))
        caption_style = ParagraphStyle("CortexCaption", parent=styles["Normal"],
                                        fontSize=8, leading=10, textColor=HexColor("#94a3b8"),
                                        alignment=TA_CENTER)

        story = []
        stats = pipeline_outputs.get("statistics", {})

        # ── Header ──
        story.append(Paragraph("CortexIQ EEG Analysis Report", title_style))
        story.append(Spacer(1, 3*mm))
        study_name = study_info.get("name", "Untitled Study")
        header_data = [
            ["Study", study_name],
            ["Date", datetime.now().strftime('%Y-%m-%d %H:%M')],
            ["Modality", study_info.get('modality', 'EEG')],
        ]
        ht = Table(header_data, colWidths=[30*mm, 130*mm])
        ht.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#10B981")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(ht)
        story.append(Spacer(1, 6*mm))

        # ── 1. Study Parameters ──
        story.append(Paragraph("1. Study Parameters", heading_style))
        desc_global = stats.get("descriptive", {}).get("global", {})
        study_details = [
            ["Parameter", "Value"],
            ["Channels", str(study_info.get("n_channels", "N/A"))],
            ["Sampling Rate", f"{study_info.get('sfreq', 'N/A')} Hz"],
            ["Duration", f"{study_info.get('duration_sec', 'N/A')} sec"],
            ["Subjects", str(study_info.get("subject_count", 1))],
            ["Total Samples", f"{desc_global.get('n_samples', 'N/A'):,}" if isinstance(desc_global.get('n_samples'), int) else str(desc_global.get('n_samples', 'N/A'))],
            ["Conditions", study_info.get("conditions", "N/A")],
        ]
        t = Table(study_details, colWidths=[50*mm, 110*mm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0D9488")),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#0f172a"), HexColor("#1e293b")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 6*mm))

        # ── 2. Descriptive Statistics (Global) ──
        story.append(Paragraph("2. Descriptive Statistics (Global)", heading_style))
        if desc_global:
            desc_table = [
                ["Statistic", "Value"],
                ["Global Mean", f"{desc_global.get('global_mean_uV', 0):.4f} µV"],
                ["Global Std Dev", f"{desc_global.get('global_std_uV', 0):.4f} µV"],
                ["Global Min", f"{desc_global.get('global_min_uV', 0):.4f} µV"],
                ["Global Max", f"{desc_global.get('global_max_uV', 0):.4f} µV"],
                ["Global Median", f"{desc_global.get('global_median_uV', 0):.4f} µV"],
                ["Total Variance", f"{desc_global.get('total_variance_uV2', 0):.4f} µV²"],
            ]
            dt = Table(desc_table, colWidths=[50*mm, 110*mm])
            dt.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#10B981")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#0f172a"), HexColor("#1e293b")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(dt)
        story.append(Spacer(1, 6*mm))

        # ── 3. Per-Channel Descriptive Statistics ──
        ch_desc = stats.get("descriptive", {}).get("channels", [])
        if ch_desc:
            story.append(Paragraph("3. Per-Channel Descriptive Statistics", heading_style))
            ch_header = ["Channel", "Mean (µV)", "Std (µV)", "Min (µV)", "Max (µV)", "Median (µV)", "Skew", "Kurt"]
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
            ct = Table(ch_rows, colWidths=[22*mm, 20*mm, 20*mm, 20*mm, 20*mm, 22*mm, 18*mm, 18*mm])
            ct.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0D9488")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#0f172a"), HexColor("#1e293b")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]))
            story.append(ct)
            story.append(Spacer(1, 6*mm))

        # ── 4. Cross-Channel Correlation ──
        corr = stats.get("descriptive", {}).get("cross_channel_correlation", {})
        if corr and corr.get("mean_r") is not None:
            story.append(Paragraph("4. Cross-Channel Correlation", heading_style))
            story.append(Paragraph(f"Mean Pearson r = <b>{corr['mean_r']:.4f}</b>", body_style))
            story.append(Spacer(1, 4*mm))

        # ── 5. Band Power Analysis ──
        band_analysis = stats.get("band_analysis", {})
        if band_analysis:
            story.append(Paragraph("5. Band Power Analysis", heading_style))
            bp_header = ["Band", "Power (V²/Hz)", "Relative Power (%)", "Power (dB)"]
            bp_rows = [bp_header]
            for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                if band in band_analysis:
                    b = band_analysis[band]
                    bp_rows.append([
                        band.capitalize(),
                        f"{b['power_V2_Hz']:.4e}",
                        f"{b['relative_power_pct']:.2f}%",
                        f"{b.get('power_dB', 'N/A')}" if b.get('power_dB') is not None else "N/A",
                    ])
            if "total_power_V2_Hz" in band_analysis:
                bp_rows.append(["Total", f"{band_analysis['total_power_V2_Hz']:.4e}", "100.00%", ""])
            bpt = Table(bp_rows, colWidths=[30*mm, 45*mm, 40*mm, 35*mm])
            bpt.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#10B981")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#0f172a"), HexColor("#1e293b")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]))
            story.append(bpt)
            story.append(Spacer(1, 6*mm))

        # ── 6. ERP Analysis ──
        erp_analysis = stats.get("erp_analysis", {})
        if erp_analysis:
            story.append(Paragraph("6. ERP Analysis", heading_style))
            erp_table = [
                ["Parameter", "Value"],
                ["Peak Channel", erp_analysis.get("peak_channel", "N/A")],
                ["Peak Latency", f"{erp_analysis.get('peak_latency_ms', 0):.2f} ms"],
                ["Peak Amplitude", f"{erp_analysis.get('peak_amplitude_uV', 0):.2f} µV"],
            ]
            et = Table(erp_table, colWidths=[50*mm, 110*mm])
            et.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0D9488")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#0f172a"), HexColor("#1e293b")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(et)
            story.append(Spacer(1, 6*mm))

        # ── 7. Epoch Analysis ──
        epoch_analysis = stats.get("epoch_analysis", {})
        if epoch_analysis:
            story.append(Paragraph("7. Epoch Analysis", heading_style))
            ep_table = [
                ["Parameter", "Value"],
                ["Number of Epochs", str(epoch_analysis.get("n_epochs", "N/A"))],
                ["Epoch Duration", f"{epoch_analysis.get('epoch_duration_sec', 0):.3f} sec"],
                ["Time Range", f"{epoch_analysis.get('tmin_sec', 0):.3f} to {epoch_analysis.get('tmax_sec', 0):.3f} sec"],
                ["Total Epoch Time", f"{epoch_analysis.get('total_epoch_time_sec', 0):.3f} sec"],
            ]
            ept = Table(ep_table, colWidths=[50*mm, 110*mm])
            ept.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#10B981")),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#0f172a"), HexColor("#1e293b")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ept)
            story.append(Spacer(1, 6*mm))

        # ── 8. Pipeline Steps ──
        steps = pipeline_outputs.get("steps", {})
        if steps:
            story.append(Paragraph("8. Pipeline Steps", heading_style))
            for step_id, step_data in steps.items():
                if isinstance(step_data, dict):
                    name = step_data.get("name", f"Step {step_id}")
                    summary = step_data.get("summary", "")
                    status = step_data.get("status", "")
                    icon = "OK" if status == "complete" else "FAIL"
                    story.append(Paragraph(f"{icon} <b>{name}</b>: {summary}", body_style))
            story.append(Spacer(1, 6*mm))

        # ── Figures ──
        fig_count = 0
        for fig_name, fig_path in figures.items():
            if os.path.exists(fig_path):
                fig_count += 1
                story.append(Paragraph(f"Figure {fig_count}: {fig_name.upper()}", heading_style))
                try:
                    img = Image(fig_path, width=160*mm, height=90*mm)
                    story.append(img)
                    story.append(Paragraph(f"Figure {fig_count}. {fig_name.upper()} plot.", caption_style))
                    story.append(Spacer(1, 5*mm))
                except Exception:
                    story.append(Paragraph(f"(Figure {fig_name} could not be embedded)", body_style))

        # ── AI Interpretation (brief) ──
        if interpretation:
            story.append(Paragraph("Interpretation", heading_style))
            story.append(Paragraph(interpretation[:500] + ("..." if len(interpretation) > 500 else ""), body_style))
            story.append(Spacer(1, 4*mm))

        # ── Methods ──
        if methods:
            story.append(Paragraph("Methods", heading_style))
            story.append(Paragraph(methods, body_style))

        # Build PDF
        doc.build(story)

        # Save comprehensive CSV
        csv_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
        self._save_csv(csv_path, stats, pipeline_outputs)

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

    def _save_csv(self, path: str, stats: dict, outputs: dict):
        """Save comprehensive numerical results to CSV with labeled sections."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Section 1: Global Descriptive Statistics
            writer.writerow(["=== GLOBAL DESCRIPTIVE STATISTICS ==="])
            writer.writerow(["Statistic", "Value"])
            desc_global = stats.get("descriptive", {}).get("global", {})
            for key, val in desc_global.items():
                label = key.replace("_", " ").replace("uV", "µV").replace("V2", "V²").title()
                writer.writerow([label, val])
            writer.writerow([])

            # Section 2: Per-Channel Descriptive Statistics
            ch_desc = stats.get("descriptive", {}).get("channels", [])
            if ch_desc:
                writer.writerow(["=== PER-CHANNEL DESCRIPTIVE STATISTICS ==="])
                writer.writerow(["Channel", "Mean (µV)", "Std (µV)", "Min (µV)", "Max (µV)",
                                 "Median (µV)", "Q25 (µV)", "Q75 (µV)", "IQR (µV)",
                                 "Skewness", "Kurtosis", "RMS (µV)", "Variance (µV²)"])
                for ch in ch_desc:
                    writer.writerow([
                        ch["channel"], ch["mean_uV"], ch["std_uV"], ch["min_uV"], ch["max_uV"],
                        ch["median_uV"], ch["q25_uV"], ch["q75_uV"], ch["iqr_uV"],
                        ch["skewness"], ch["kurtosis"], ch["rms_uV"], ch["variance_uV2"],
                    ])
                writer.writerow([])

            # Section 3: Cross-Channel Correlation Matrix
            corr = stats.get("descriptive", {}).get("cross_channel_correlation", {})
            if corr and corr.get("matrix"):
                writer.writerow(["=== CROSS-CHANNEL CORRELATION (Pearson r) ==="])
                writer.writerow([""] + corr["channels"])
                for i, row in enumerate(corr["matrix"]):
                    writer.writerow([corr["channels"][i]] + row)
                writer.writerow(["Mean r", corr.get("mean_r", "")])
                writer.writerow([])

            # Section 4: Band Power Analysis
            band_analysis = stats.get("band_analysis", {})
            if band_analysis:
                writer.writerow(["=== BAND POWER ANALYSIS ==="])
                writer.writerow(["Band", "Power (V²/Hz)", "Relative Power (%)", "Power (dB)"])
                for band in ["delta", "theta", "alpha", "beta", "gamma"]:
                    if band in band_analysis:
                        b = band_analysis[band]
                        writer.writerow([band, b["power_V2_Hz"], b["relative_power_pct"],
                                        b.get("power_dB", "")])
                if "total_power_V2_Hz" in band_analysis:
                    writer.writerow(["Total", band_analysis["total_power_V2_Hz"], "100.00", ""])
                writer.writerow([])

                # Per-channel band powers
                per_ch = band_analysis.get("per_channel", [])
                if per_ch:
                    writer.writerow(["=== PER-CHANNEL BAND POWERS ==="])
                    bands = ["delta", "theta", "alpha", "beta", "gamma"]
                    writer.writerow(["Channel"] + [f"{b} (V²/Hz)" for b in bands])
                    for row in per_ch:
                        writer.writerow([row["channel"]] + [row.get(f"{b}_power", "") for b in bands])
                    writer.writerow([])

            # Section 5: ERP Analysis
            erp_analysis = stats.get("erp_analysis", {})
            if erp_analysis:
                writer.writerow(["=== ERP ANALYSIS ==="])
                writer.writerow(["Parameter", "Value"])
                for key, val in erp_analysis.items():
                    label = key.replace("_", " ").title()
                    writer.writerow([label, val])
                writer.writerow([])

            # Section 6: Epoch Analysis
            epoch_analysis = stats.get("epoch_analysis", {})
            if epoch_analysis:
                writer.writerow(["=== EPOCH ANALYSIS ==="])
                writer.writerow(["Parameter", "Value"])
                for key, val in epoch_analysis.items():
                    label = key.replace("_", " ").title()
                    writer.writerow([label, val])
                writer.writerow([])

            # Section 7: Pipeline Step Results
            steps = outputs.get("steps", {})
            if steps:
                writer.writerow(["=== PIPELINE STEP RESULTS ==="])
                writer.writerow(["Step ID", "Name", "Status", "Summary"])
                for sid, sdata in steps.items():
                    if isinstance(sdata, dict):
                        writer.writerow([sid, sdata.get("name", ""), sdata.get("status", ""),
                                        sdata.get("summary", "")])
