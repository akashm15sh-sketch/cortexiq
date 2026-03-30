# CortexIQ EEG Prototype

AI-powered EEG analysis engine for brain signal processing. Locally-hosted, uses Claude Opus for intelligent pipeline generation and MNE-Python for real EEG signal analysis.

## Prerequisites
- Python 3.11+
- pip

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your Anthropic API key to .env
python app.py
```
Opens at **http://localhost:7860**

## Demo Licence Keys

| Key | Tier | Max Logins |
|-----|------|-----------|
| `CORTEX-DEMO-2025-PITCH` | Researcher | 5 |
| `CORTEX-LAB-IIT-001` | Lab | 5 |
| `CORTEX-TRIAL-FREE-01` | Explorer | 5 |

## Supported EEG Formats
`.edf`, `.bdf`, `.fif`, `.set`, `.vhdr`, `.csv`, `.tsv`, `.npy`

## Features
- **JWT licence-key authentication** with login counters
- **Claude Opus AI assistant** — domain-locked to EEG analysis
- **MNE-Python pipeline** — real signal processing (filter, ICA, PSD, ERP)
- **Pauseable execution** — pause/resume/stop mid-pipeline
- **PDF reports** with embedded figures and AI interpretation
- **Interactive Plotly charts** — PSD, ERP, topographic maps

> **Note:** fMRI and MEG modalities are marked *Under Construction* for future releases.
