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
            elif ext == ".xdf":
                raw = self._load_xdf(filepath)
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
            import traceback
            traceback.print_exc()
            return None, FileInfo(error=f"Load error: {str(e)}")

    def _load_xdf(self, filepath: str) -> mne.io.RawArray:
        """Load XDF file using pyxdf."""
        import pyxdf
        streams, header = pyxdf.load_xdf(filepath)
        
        # Find the EEG stream (highest sfreq usually, or labeled EEG)
        eeg_stream = None
        for s in streams:
            if s['info']['type'][0].lower() == 'eeg':
                eeg_stream = s
                break
        
        # Fallback: largest numeric stream
        if eeg_stream is None:
            max_ch = -1
            for s in streams:
                if int(s['info']['channel_count'][0]) > max_ch:
                    max_ch = int(s['info']['channel_count'][0])
                    eeg_stream = s

        if eeg_stream is None:
            raise ValueError("No valid EEG streams found in XDF file.")

        data = eeg_stream['time_series'].T  # (n_channels, n_times)
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        
        # Get channel names if available
        ch_names = []
        try:
            desc = eeg_stream['info']['desc'][0]
            if desc and 'channels' in desc:
                channels = desc['channels'][0]['channel']
                for ch in channels:
                    ch_names.append(ch['label'][0])
        except Exception:
            pass
            
        if len(ch_names) != data.shape[0]:
            ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]

        # Scale if necessary (MNE expects Volts)
        # Check unit (standard XDF units are microvolts usually)
        if np.abs(data).max() > 1.0:
            data = data * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        return mne.io.RawArray(data, info, verbose=False)

    def _load_tabular(self, filepath: str, ext: str, sfreq: float = None) -> mne.io.RawArray:
        """Load CSV/TSV into MNE RawArray."""
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(filepath, sep=sep)

        # Try to load sfreq from JSON sidecar
        if sfreq is None:
            sfreq = self._try_json_sidecar(filepath)
        if sfreq is None:
            sfreq = 256.0  # Default

        # Detect if column names are numeric (CSV has no header)
        all_numeric_headers = all(
            self._is_numeric_str(str(c)) for c in df.columns
        )
        if all_numeric_headers:
            df_no_header = pd.read_csv(filepath, sep=sep, header=None)
            first_col = str(df_no_header.iloc[0, 0]).lower().strip()
            if first_col in ("time", "timestamp", "sample", "index", ""):
                df_no_header = df_no_header.iloc[1:]
            df_no_header = df_no_header.apply(pd.to_numeric, errors="coerce")
            df_no_header = df_no_header.dropna(axis=1, how="all")
            n_cols = len(df_no_header.columns)
            if n_cols > 1:
                first_vals = df_no_header.iloc[:5, 0].values
                is_time = all(
                    isinstance(v, (int, float)) and (i == 0 or v > first_vals[i-1])
                    for i, v in enumerate(first_vals) if not np.isnan(v)
                )
                if is_time and n_cols > 2:
                    df_no_header = df_no_header.iloc[:, 1:]
            df = df_no_header.reset_index(drop=True)
            df.columns = [f"Ch{i+1}" for i in range(len(df.columns))]

        eeg_cols = []
        other_cols = []
        for col in df.columns:
            if col.lower().strip() in EEG_CHANNEL_PATTERNS or col.lower().startswith("eeg"):
                eeg_cols.append(col)
            elif col.lower() in ("time", "timestamp", "sample", "index"):
                continue
            else:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                    other_cols.append(col)

        use_cols = eeg_cols if eeg_cols else other_cols
        if not use_cols:
            use_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        if not use_cols:
            raise ValueError("No numeric EEG channels found in the file.")

        data = df[use_cols].values.T
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
            data = data.T

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
