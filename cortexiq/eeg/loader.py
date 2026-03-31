"""EEG file loader — auto-detect format, load into MNE Raw."""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
import mne


@dataclass
class FileInfo:
    format_name: str = "Unknown"
    n_channels: int = 0
    sfreq: float = 0.0
    duration_sec: float = 0.0
    channel_names: List[str] = field(default_factory=list)
    bad_channels: List[str] = field(default_factory=list)
    error: Optional[str] = None


# Common EEG channel name patterns
EEG_CHANNEL_PATTERNS = [
    "fp1", "fp2", "f3", "f4", "c3", "c4", "p3", "p4", "o1", "o2",
    "f7", "f8", "t3", "t4", "t5", "t6", "fz", "cz", "pz", "oz",
    "af3", "af4", "fc1", "fc2", "fc5", "fc6", "cp1", "cp2", "cp5", "cp6",
    "po3", "po4", "af7", "af8", "ft7", "ft8", "tp7", "tp8", "po7", "po8",
    "fpz", "afz", "fcz", "cpz", "poz",
    "t7", "t8", "p7", "p8",
    "eeg", "eog", "emg",
]


class EEGLoader:
    """Load EEG files into MNE Raw objects."""

    def load(self, filepath: str, sfreq: float = None) -> tuple:
        """Load an EEG file. Returns (mne.io.Raw | mne.Epochs, FileInfo)."""
        ext = os.path.splitext(filepath)[1].lower()
        # Handle .nii.gz
        if filepath.lower().endswith(".nii.gz"):
            ext = ".nii.gz"

        try:
            if ext == ".edf":
                raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            elif ext == ".bdf":
                raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
            elif ext == ".fif":
                raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
            elif ext == ".set":
                raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
            elif ext == ".vhdr":
                raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
            elif ext in (".csv", ".tsv"):
                raw = self._load_tabular(filepath, ext, sfreq)
            elif ext == ".npy":
                raw = self._load_numpy(filepath, sfreq)
            else:
                return None, FileInfo(error=f"Unsupported format: {ext}")

            info = FileInfo(
                format_name=ext.upper().replace(".", ""),
                n_channels=len(raw.ch_names),
                sfreq=raw.info["sfreq"],
                duration_sec=raw.n_times / raw.info["sfreq"],
                channel_names=list(raw.ch_names),
                bad_channels=list(raw.info.get("bads", [])),
            )
            return raw, info

        except Exception as e:
            return None, FileInfo(error=f"Load error: {str(e)}")

    def _load_tabular(self, filepath: str, ext: str, sfreq: float = None) -> mne.io.RawArray:
        """Load CSV/TSV into MNE RawArray."""
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(filepath, sep=sep)

        # Try to load sfreq from JSON sidecar
        if sfreq is None:
            sfreq = self._try_json_sidecar(filepath)
        if sfreq is None:
            sfreq = 256.0  # Default

        # Detect if column names are numeric (CSV has no header — first data row used as header)
        all_numeric_headers = all(
            self._is_numeric_str(str(c)) for c in df.columns
        )
        if all_numeric_headers:
            # Reload without header, prepend the "header" row back as data
            df_no_header = pd.read_csv(filepath, sep=sep, header=None)
            # Check if first row looks like a time column
            first_col = str(df_no_header.iloc[0, 0]).lower().strip()
            if first_col in ("time", "timestamp", "sample", "index", ""):
                df_no_header = df_no_header.iloc[1:]  # Skip time label row
            else:
                # The header row is actually data — keep it
                pass
            df_no_header = df_no_header.apply(pd.to_numeric, errors="coerce")
            df_no_header = df_no_header.dropna(axis=1, how="all")
            n_cols = len(df_no_header.columns)
            # Drop first column if it looks like time/index
            if n_cols > 1:
                first_vals = df_no_header.iloc[:5, 0].values
                is_time = all(
                    isinstance(v, (int, float)) and (i == 0 or v > first_vals[i-1])
                    for i, v in enumerate(first_vals) if not np.isnan(v)
                )
                if is_time and n_cols > 2:
                    df_no_header = df_no_header.iloc[:, 1:]
                    n_cols -= 1
            df = df_no_header.reset_index(drop=True)
            df.columns = [f"Ch{i+1}" for i in range(len(df.columns))]

        # Detect EEG columns
        eeg_cols = []
        other_cols = []
        for col in df.columns:
            if col.lower().strip() in EEG_CHANNEL_PATTERNS or col.lower().startswith("eeg"):
                eeg_cols.append(col)
            elif col.lower() in ("time", "timestamp", "sample", "index"):
                continue  # Skip time columns
            else:
                # Try as numeric channel
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                    other_cols.append(col)

        use_cols = eeg_cols if eeg_cols else other_cols
        if not use_cols:
            use_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        if not use_cols:
            raise ValueError("No numeric EEG channels found in the file.")

        data = df[use_cols].values.T  # (n_channels, n_times)

        # Scale to volts if data looks like microvolts
        if np.abs(data).max() > 1.0:
            data = data * 1e-6

        ch_types = ["eeg"] * len(use_cols)
        info = mne.create_info(ch_names=use_cols, sfreq=sfreq, ch_types=ch_types)
        return mne.io.RawArray(data, info, verbose=False)

    @staticmethod
    def _is_numeric_str(s: str) -> bool:
        """Check if a string represents a number."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _load_numpy(self, filepath: str, sfreq: float = None) -> mne.io.RawArray:
        """Load numpy array into MNE RawArray."""
        data = np.load(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] > data.shape[1]:
            data = data.T  # Ensure (n_channels, n_times)

        if sfreq is None:
            sfreq = 256.0

        ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
        ch_types = ["eeg"] * data.shape[0]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        return mne.io.RawArray(data, info, verbose=False)

    def _try_json_sidecar(self, filepath: str) -> float | None:
        """Try to read sampling frequency from a JSON sidecar file."""
        import json
        base = os.path.splitext(filepath)[0]
        for suffix in [".json", "_eeg.json"]:
            json_path = base + suffix
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        data = json.load(f)
                    return float(data.get("SamplingFrequency", data.get("sfreq", data.get("sampling_rate", 0))))
                except Exception:
                    pass
        return None
