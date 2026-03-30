"""Pauseable EEG analysis pipeline with MNE-Python."""
import threading
import time
import traceback
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from ..config import RESULTS_DIR


class EEGPipeline:
    """MNE-Python pipeline with pause/resume/stop support."""

    def __init__(self):
        self.status = "idle"  # idle, running, paused, complete, failed
        self.current_step = 0
        self.steps = []
        self.raw = None
        self.epochs = None
        self.step_outputs = {}
        self.figures = {}
        self.results = {}  # Separate store for computed results (band_powers, erp_peak)
        self.pause_event = threading.Event()
        self.pause_event.set()  # Start unpaused
        self.stop_flag = False
        self.log_messages = []
        self._thread = None

    def _log(self, msg):
        self.log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.log_messages) > 50:
            self.log_messages = self.log_messages[-30:]

    def run(self, steps: list, raw: mne.io.BaseRaw, progress_callback=None):
        """Start pipeline execution in a background thread."""
        self.steps = steps
        self.raw = raw.copy()
        self.epochs = None
        self.step_outputs = {}
        self.figures = {}
        self.results = {}
        self.current_step = 0
        self.stop_flag = False
        self.pause_event.set()
        self.status = "running"
        self.log_messages = []

        os.makedirs(RESULTS_DIR, exist_ok=True)

        def _execute():
            try:
                for i, step in enumerate(self.steps):
                    # Check stop
                    if self.stop_flag:
                        self.status = "stopped"
                        self._log("Pipeline stopped by user.")
                        if progress_callback:
                            progress_callback(i, "stopped", "Pipeline stopped.")
                        return

                    # Check pause (blocks here if paused)
                    self.pause_event.wait()

                    if self.stop_flag:
                        self.status = "stopped"
                        self._log("Pipeline stopped by user.")
                        return

                    self.current_step = i
                    step_name = step.get("name", f"Step {i+1}")
                    tool = step.get("tool", "unknown")
                    params = step.get("parameters", {})

                    self._log(f"▶ Running: {step_name} ({tool})")
                    if progress_callback:
                        progress_callback(i, "running", f"Running {step_name}...")

                    try:
                        result = self._execute_step(step_name, tool, params)
                        self.step_outputs[i] = {"status": "complete", "summary": result, "name": step_name}
                        self._log(f"✓ Complete: {step_name} — {result}")
                        if progress_callback:
                            progress_callback(i, "complete", result)
                    except Exception as e:
                        error_msg = f"Failed: {str(e)}"
                        self.step_outputs[i] = {"status": "failed", "summary": error_msg, "name": step_name}
                        self._log(f"✗ {step_name}: {error_msg}")
                        if progress_callback:
                            progress_callback(i, "failed", error_msg)
                        # Continue to next step instead of stopping entirely

                self.status = "complete"
                self._log("Pipeline complete.")
                if progress_callback:
                    progress_callback(len(self.steps), "complete", "All steps finished.")
            except Exception as e:
                self.status = "failed"
                self._log(f"Pipeline error: {str(e)}")
                traceback.print_exc()

        self._thread = threading.Thread(target=_execute, daemon=True)
        self._thread.start()

    def pause(self):
        self.pause_event.clear()
        self.status = "paused"
        self._log(f"⏸ Paused after step {self.current_step + 1}")

    def resume(self):
        self.pause_event.set()
        self.status = "running"
        self._log("▶ Resumed")

    def stop(self):
        self.stop_flag = True
        self.pause_event.set()  # Unblock if paused
        self._log("⏹ Stop requested")

    def reset(self):
        """Reset pipeline state for a fresh run."""
        self.status = "idle"
        self.current_step = 0
        self.steps = []
        self.raw = None
        self.epochs = None
        self.step_outputs = {}
        self.figures = {}
        self.results = {}
        self.pause_event.set()
        self.stop_flag = False
        self.log_messages = []
        self._thread = None
        self._psd_freqs = None
        self._psd_data = None

    @staticmethod
    def get_step_code(name: str, tool: str, params: dict) -> str:
        """Generate example Python code for a given pipeline step."""
        nl = name.lower()
        tl = tool.lower()
        if "filter" in nl or "filter" in tl:
            l_freq = params.get("l_freq", 0.1)
            h_freq = params.get("h_freq", 100.0)
            notch = params.get("notch", 50.0)
            return (
                f"import mne\n\n"
                f"# Band-pass filter\n"
                f"raw.filter(l_freq={l_freq}, h_freq={h_freq}, verbose=False)\n\n"
                f"# Notch filter to remove power line noise\n"
                f"raw.notch_filter(freqs={notch}, verbose=False)"
            )
        elif "bad" in nl or "ransac" in tl or "bad_channel" in nl:
            threshold = params.get("z_threshold", 3.0)
            return (
                f"import numpy as np\n\n"
                f"# Detect bad channels by variance thresholding\n"
                f"data = raw.get_data()\n"
                f"variances = np.var(data, axis=1)\n"
                f"mean_var = np.mean(variances)\n"
                f"std_var = np.std(variances)\n"
                f"threshold = {threshold}\n\n"
                f"bad_mask = np.abs(variances - mean_var) > threshold * std_var\n"
                f"bad_chs = [raw.ch_names[i] for i in np.where(bad_mask)[0]]\n"
                f"raw.info['bads'] = bad_chs"
            )
        elif "reference" in nl or "set_eeg_reference" in tl:
            ref = params.get("ref", "average")
            return (
                f"# Re-reference EEG data\n"
                f"raw.set_eeg_reference('{ref}', verbose=False)"
            )
        elif "ica" in nl or "ica" in tl:
            n = params.get("n_components", 15)
            return (
                f"from mne.preprocessing import ICA\n\n"
                f"# Fit ICA for artifact removal\n"
                f"ica = ICA(n_components={n}, method='fastica',\n"
                f"          random_state=42, verbose=False)\n"
                f"ica.fit(raw, verbose=False)\n\n"
                f"# Find and remove EOG artifacts\n"
                f"eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)\n"
                f"ica.exclude = eog_indices\n"
                f"ica.apply(raw, verbose=False)"
            )
        elif "epoch" in nl or "epoch" in tl:
            dur = params.get("epoch_duration", 2.0)
            return (
                f"import mne\n\n"
                f"# Create epochs from events (or fixed-length)\n"
                f"try:\n"
                f"    events = mne.find_events(raw, verbose=False)\n"
                f"    epochs = mne.Epochs(raw, events, preload=True, verbose=False)\n"
                f"except Exception:\n"
                f"    epochs = mne.make_fixed_length_epochs(\n"
                f"        raw, duration={dur}, preload=True, verbose=False\n"
                f"    )"
            )
        elif "psd" in nl or "power" in nl or "spectral" in nl or "compute_psd" in tl:
            fmin = params.get("fmin", 1.0)
            fmax = params.get("fmax", 45.0)
            return (
                f"import numpy as np\n\n"
                f"# Compute Power Spectral Density (Welch method)\n"
                f"psd = raw.compute_psd(method='welch',\n"
                f"                      fmin={fmin}, fmax={fmax}, verbose=False)\n\n"
                f"# Extract band powers\n"
                f"bands = {{'delta': (1, 4), 'theta': (4, 8),\n"
                f"         'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}}\n"
                f"freqs = psd.freqs\n"
                f"psd_data = psd.get_data()\n\n"
                f"for band, (flo, fhi) in bands.items():\n"
                f"    idx = np.logical_and(freqs >= flo, freqs <= fhi)\n"
                f"    print(f'{{band}}: {{np.mean(psd_data[:, idx]):.2e}} V^2/Hz')"
            )
        elif "erp" in nl or "evoked" in nl:
            return (
                f"# Compute Event-Related Potential\n"
                f"evoked = epochs.average()\n\n"
                f"# Find peak amplitude\n"
                f"ch, latency, amplitude = evoked.get_peak(return_amplitude=True)\n"
                f"print(f'Peak: {{latency*1000:.1f}} ms, {{amplitude*1e6:.2f}} uV on {{ch}}')\n\n"
                f"# Plot ERP butterfly\n"
                f"evoked.plot(show=True)"
            )
        else:
            return f"# {name}\n# MNE tool: {tool}\n# Parameters: {params}"

    def _execute_step(self, name: str, tool: str, params: dict) -> str:
        """Route to the appropriate MNE function."""
        tool_lower = tool.lower()
        name_lower = name.lower()

        if "filter" in name_lower or "filter" in tool_lower:
            return self._filter_data(params)
        elif "bad" in name_lower or "ransac" in tool_lower or "bad_channel" in name_lower:
            return self._detect_bad_channels(params)
        elif "reference" in name_lower or "set_eeg_reference" in tool_lower:
            return self._set_reference(params)
        elif "ica" in name_lower or "ica" in tool_lower:
            return self._run_ica(params)
        elif "epoch" in name_lower or "epoch" in tool_lower:
            return self._epoch_data(params)
        elif "psd" in name_lower or "power" in name_lower or "spectral" in name_lower or "compute_psd" in tool_lower:
            return self._compute_psd(params)
        elif "erp" in name_lower or "evoked" in name_lower or "average" in tool_lower:
            return self._compute_erp(params)
        else:
            time.sleep(1)  # Simulate unknown step
            return f"Step '{name}' completed (simulated)."

    def _filter_data(self, params: dict) -> str:
        l_freq = params.get("l_freq", 0.1)
        h_freq = params.get("h_freq", 100.0)
        notch = params.get("notch", 50.0)

        self.raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        if notch:
            self.raw.notch_filter(freqs=notch, verbose=False)
        return f"Bandpass filtered {l_freq}–{h_freq} Hz. Notch at {notch} Hz applied."

    def _detect_bad_channels(self, params: dict) -> str:
        # Use variance-based detection
        data = self.raw.get_data()
        variances = np.var(data, axis=1)
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        threshold = params.get("z_threshold", 3.0)

        bad_mask = np.abs(variances - mean_var) > threshold * std_var
        bad_chs = [self.raw.ch_names[i] for i in np.where(bad_mask)[0]]

        self.raw.info["bads"] = bad_chs
        return f"Found {len(bad_chs)} bad channels: {', '.join(bad_chs) if bad_chs else 'none'}."

    def _set_reference(self, params: dict) -> str:
        ref = params.get("ref", "average")
        if ref == "average":
            self.raw.set_eeg_reference("average", verbose=False)
        else:
            self.raw.set_eeg_reference(ref_channels=[ref], verbose=False)
        return f"Re-referenced to {ref}."

    def _run_ica(self, params: dict) -> str:
        n_components = min(params.get("n_components", 15), len(self.raw.ch_names) - 1)
        ica = mne.preprocessing.ICA(n_components=n_components, method="fastica", random_state=42, verbose=False)
        ica.fit(self.raw, verbose=False)

        # Try to find EOG artifacts
        n_excluded = 0
        try:
            eog_indices, eog_scores = ica.find_bads_eog(self.raw, verbose=False)
            ica.exclude = eog_indices
            n_excluded = len(eog_indices)
        except Exception:
            pass

        ica.apply(self.raw, verbose=False)

        # Save ICA component figure
        try:
            fig = ica.plot_components(show=False)
            if isinstance(fig, list):
                fig = fig[0]
            fig_path = os.path.join(RESULTS_DIR, "ica_components.png")
            fig.savefig(fig_path, dpi=100, bbox_inches="tight", facecolor="#07080f")
            plt.close(fig)
            self.figures["ica"] = fig_path
        except Exception:
            pass

        return f"ICA: {n_components} components fitted. {n_excluded} artifact components removed."

    def _epoch_data(self, params: dict) -> str:
        duration = params.get("epoch_duration", 2.0)
        try:
            events = mne.find_events(self.raw, verbose=False)
            self.epochs = mne.Epochs(self.raw, events, preload=True, verbose=False)
            return f"Created {len(self.epochs)} epochs from {len(events)} events."
        except Exception:
            self.epochs = mne.make_fixed_length_epochs(self.raw, duration=duration, preload=True, verbose=False)
            return f"Created {len(self.epochs)} fixed-length epochs ({duration}s each)."

    def _compute_psd(self, params: dict) -> str:
        fmin = params.get("fmin", 1.0)
        fmax = params.get("fmax", 45.0)

        psd = self.raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)

        # Calculate band powers
        bands = {
            "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
            "beta": (13, 30), "gamma": (30, 45)
        }
        band_powers = {}
        freqs = psd.freqs
        psd_data = psd.get_data()  # (n_channels, n_freqs)
        self._psd_freqs = freqs
        self._psd_data = psd_data

        for band_name, (flo, fhi) in bands.items():
            idx = np.logical_and(freqs >= flo, freqs <= fhi)
            if idx.any():
                band_powers[band_name] = float(np.mean(psd_data[:, idx]))

        self.results["band_powers"] = band_powers

        # Save PSD figure
        try:
            fig = psd.plot(show=False)
            fig_path = os.path.join(RESULTS_DIR, "psd_plot.png")
            fig.savefig(fig_path, dpi=100, bbox_inches="tight", facecolor="#07080f")
            plt.close(fig)
            self.figures["psd"] = fig_path
        except Exception:
            pass

        bp_str = ", ".join([f"{k}: {v:.2e}" for k, v in band_powers.items()])
        return f"PSD computed. Band powers — {bp_str}"

    def _compute_erp(self, params: dict) -> str:
        if self.epochs is None:
            raise RuntimeError("No epochs available — run epoching first.")

        evoked = self.epochs.average()

        # Save ERP figure
        try:
            fig = evoked.plot(show=False)
            fig_path = os.path.join(RESULTS_DIR, "erp_butterfly.png")
            fig.savefig(fig_path, dpi=100, bbox_inches="tight", facecolor="#07080f")
            plt.close(fig)
            self.figures["erp"] = fig_path
        except Exception:
            pass

        # Get peak
        try:
            ch_name, latency, amplitude = evoked.get_peak(return_amplitude=True)
            self.results["erp_peak"] = {"channel": ch_name, "latency_ms": latency * 1000, "amplitude_uV": amplitude * 1e6}
            return f"ERP computed. Peak at {latency*1000:.1f} ms ({amplitude*1e6:.2f} µV) on {ch_name}."
        except Exception:
            return "ERP computed. Peak detection failed — check epoch quality."

    def get_results_summary(self) -> dict:
        """Get a summary of all completed steps and results."""
        stats = self._compute_statistics()
        return {
            "steps": self.step_outputs,
            "figures": self.figures,
            "band_powers": self.results.get("band_powers", {}),
            "erp_peak": self.results.get("erp_peak", {}),
            "status": self.status,
            "log": self.log_messages[-15:],
            "statistics": stats,
        }

    def _compute_statistics(self) -> dict:
        """Compute descriptive and inferential statistics from current data."""
        import scipy.stats as sp_stats
        result = {"descriptive": {}, "band_analysis": {}, "erp_analysis": {}}

        if self.raw is None:
            return result

        try:
            sfreq = self.raw.info["sfreq"]
            ch_names = self.raw.ch_names
            n_ch = len(ch_names)
            n_times_total = self.raw.n_times
            duration = n_times_total / sfreq

            # Cap to first 60s for stats to avoid slow computation
            cap_samples = min(n_times_total, int(60 * sfreq))
            data = self.raw.get_data(stop=cap_samples)  # (n_channels, cap_samples)
            data_uv = data * 1e6  # Convert to µV

            # ── Per-channel descriptive statistics (vectorized) ──
            ch_means = np.mean(data_uv, axis=1)
            ch_stds = np.std(data_uv, axis=1)
            ch_mins = np.min(data_uv, axis=1)
            ch_maxs = np.max(data_uv, axis=1)
            ch_medians = np.median(data_uv, axis=1)
            ch_q25 = np.percentile(data_uv, 25, axis=1)
            ch_q75 = np.percentile(data_uv, 75, axis=1)
            ch_iqr = ch_q75 - ch_q25
            ch_rms = np.sqrt(np.mean(data_uv ** 2, axis=1))
            ch_vars = np.var(data_uv, axis=1)
            ch_skew = sp_stats.skew(data_uv, axis=1)
            ch_kurt = sp_stats.kurtosis(data_uv, axis=1)

            ch_desc = []
            for i in range(n_ch):
                ch_desc.append({
                    "channel": ch_names[i],
                    "mean_uV": round(float(ch_means[i]), 4),
                    "std_uV": round(float(ch_stds[i]), 4),
                    "min_uV": round(float(ch_mins[i]), 4),
                    "max_uV": round(float(ch_maxs[i]), 4),
                    "median_uV": round(float(ch_medians[i]), 4),
                    "q25_uV": round(float(ch_q25[i]), 4),
                    "q75_uV": round(float(ch_q75[i]), 4),
                    "iqr_uV": round(float(ch_iqr[i]), 4),
                    "skewness": round(float(ch_skew[i]), 4),
                    "kurtosis": round(float(ch_kurt[i]), 4),
                    "rms_uV": round(float(ch_rms[i]), 4),
                    "variance_uV2": round(float(ch_vars[i]), 4),
                })
            result["descriptive"]["channels"] = ch_desc

            # ── Global descriptive statistics ──
            result["descriptive"]["global"] = {
                "n_channels": n_ch,
                "n_samples": n_times_total,
                "duration_sec": round(duration, 3),
                "sampling_rate_Hz": sfreq,
                "global_mean_uV": round(float(np.mean(data_uv)), 4),
                "global_std_uV": round(float(np.std(data_uv)), 4),
                "global_min_uV": round(float(np.min(data_uv)), 4),
                "global_max_uV": round(float(np.max(data_uv)), 4),
                "global_median_uV": round(float(np.median(data_uv)), 4),
                "total_variance_uV2": round(float(np.var(data_uv)), 4),
            }

            # ── Band power statistics (from stored results) ──
            bp = self.results.get("band_powers", {})
            if bp:
                total_power = sum(bp.values())
                band_analysis = {}
                for band, power in bp.items():
                    band_analysis[band] = {
                        "power_V2_Hz": float(power),
                        "relative_power_pct": round(float(power / total_power * 100), 2) if total_power > 0 else 0,
                        "power_dB": round(float(10 * np.log10(power)), 2) if power > 0 else None,
                    }
                band_analysis["total_power_V2_Hz"] = float(total_power)
                result["band_analysis"] = band_analysis

            # ── Channel-level band powers ──
            if bp and hasattr(self, '_psd_data') and hasattr(self, '_psd_freqs'):
                bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
                ch_bands = []
                for i in range(min(n_ch, len(ch_names))):
                    row = {"channel": ch_names[i]}
                    for bname, (flo, fhi) in bands.items():
                        idx = np.logical_and(self._psd_freqs >= flo, self._psd_freqs <= fhi)
                        if idx.any():
                            row[f"{bname}_power"] = round(float(np.mean(self._psd_data[i, idx])), 8)
                    ch_bands.append(row)
                result["band_analysis"]["per_channel"] = ch_bands

            # ── Epoch statistics ──
            if self.epochs is not None:
                n_epochs = len(self.epochs)
                epoch_dur = (self.epochs.times[-1] - self.epochs.times[0])
                result["epoch_analysis"] = {
                    "n_epochs": n_epochs,
                    "epoch_duration_sec": round(float(epoch_dur), 3),
                    "tmin_sec": round(float(self.epochs.times[0]), 3),
                    "tmax_sec": round(float(self.epochs.times[-1]), 3),
                    "total_epoch_time_sec": round(float(n_epochs * epoch_dur), 3),
                }

            # ── ERP statistics ──
            erp = self.results.get("erp_peak", {})
            if erp:
                result["erp_analysis"] = {
                    "peak_channel": erp.get("channel", ""),
                    "peak_latency_ms": round(float(erp.get("latency_ms", 0)), 2),
                    "peak_amplitude_uV": round(float(erp.get("amplitude_uV", 0)), 2),
                }

        except Exception as e:
            result["error"] = str(e)

        return result
