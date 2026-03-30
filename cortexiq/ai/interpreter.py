"""CortexIQ AI Interpreter — supports Anthropic Claude and Kilo Gateway."""
import os
import re
import json
import requests
from anthropic import Anthropic
from .system_prompt import CORTEXIQ_SYSTEM_PROMPT


import logging
logger = logging.getLogger(__name__)

KILO_BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"

# Map friendly model names to provider identifiers
MODEL_MAP = {
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "kilo-claude": {"provider": "kilo", "model": "anthropic/claude-sonnet-4-20250514"},
    "kilo-gpt4o": {"provider": "kilo", "model": "openai/gpt-4o"},
    "kilo-gemini": {"provider": "kilo", "model": "google/gemini-2.5-pro"},
}


def _extract_json(text: str) -> dict | None:
    """Multi-pass JSON extraction: handles markdown fences, extra text, etc."""
    cleaned = text.strip()
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip(), strict=False)
        except Exception:
            pass

    start = cleaned.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == '{':
                depth += 1
            elif cleaned[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        return json.loads(candidate, strict=False)
                    except Exception:
                        continue

    logger.error("JSON extraction total failure")
    return None


class CortexIQInterpreter:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.kilo_key = os.getenv("KILO_API_KEY")
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

    def _call_anthropic(self, messages: list, system: str, max_tokens: int = 4000) -> str:
        """Call Anthropic API directly."""
        if not self.client:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise RuntimeError("Anthropic API Key not set.")
            self.client = Anthropic(api_key=self.api_key)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()

    def _call_kilo(self, messages: list, system: str, model: str, max_tokens: int = 4000) -> str:
        """Call Kilo Gateway (OpenAI-compatible) API."""
        kilo_key = self.kilo_key or os.getenv("KILO_API_KEY")
        if not kilo_key:
            raise RuntimeError("Kilo API Key not set.")

        # Prepend system message
        full_messages = [{"role": "system", "content": system}] + messages

        headers = {
            "Authorization": f"Bearer {kilo_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        resp = requests.post(KILO_BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def interpret(self, user_prompt: str, study_context: dict, conversation_history: list = None, model: str = "claude") -> dict:
        model_config = MODEL_MAP.get(model, MODEL_MAP["claude"])
        provider = model_config["provider"]
        model_id = model_config["model"]

        # Build context string
        ctx_parts = []
        if study_context:
            ctx_parts.append(f"Study Context:")
            ctx_parts.append(f"  Study Name: {study_context.get('name', 'Untitled')}")
            ctx_parts.append(f"  Modality: {study_context.get('modality', 'EEG')}")
            ctx_parts.append(f"  Format: {study_context.get('file_format', 'unknown')}")
            ctx_parts.append(f"  Total Subjects: {study_context.get('n_subjects', 'unknown')}")
            ctx_parts.append(f"  Channels: {study_context.get('n_channels', 'unknown')}")
            ch_names = study_context.get('channel_names', [])
            if ch_names:
                if len(ch_names) > 32:
                    ctx_parts.append(f"  Channel Names (truncated): {', '.join(ch_names[:32])} ...")
                else:
                    ctx_parts.append(f"  Channel Names: {', '.join(ch_names)}")
            ctx_parts.append(f"  Sampling Rate: {study_context.get('sfreq', 'unknown')} Hz")
            ctx_parts.append(f"  Duration: {study_context.get('duration_sec', 'unknown')} sec")
            ctx_parts.append(f"  Total Recording Duration: {study_context.get('total_duration_sec', 'unknown')} sec")
            ctx_parts.append(f"  Conditions: {study_context.get('conditions', 'not specified')}")
            ctx_parts.append(f"  Reference: {study_context.get('reference', 'not specified')}")
            if study_context.get('montage'):
                ctx_parts.append(f"  Electrode Montage: {', '.join(study_context['montage'])}")
            if study_context.get('notes'):
                ctx_parts.append(f"  Notes: {study_context['notes']}")

            subjects = study_context.get('subjects', [])
            if subjects:
                ctx_parts.append(f"\nPer-Subject Details:")
                for i, subj in enumerate(subjects):
                    ctx_parts.append(f"  Subject {i+1} ({subj.get('name', 'unknown')}): "
                                     f"{subj.get('n_channels', '?')} channels, "
                                     f"{subj.get('sfreq', '?')} Hz, "
                                     f"{subj.get('duration_sec', '?')} sec, "
                                     f"format={subj.get('format', '?')}")

            data_stats = study_context.get('data_stats', [])
            if data_stats:
                ctx_parts.append(f"\nData Statistics:")
                for i, ds in enumerate(data_stats):
                    ctx_parts.append(f"  Subject {i+1} ({ds.get('name', '?')}): "
                                     f"{ds.get('n_samples', '?')} samples, "
                                     f"amplitude range: {ds.get('amplitude_uV_range', '?')} µV, "
                                     f"std: {ds.get('amplitude_uV_std', '?')} µV")
                    ch_stats = ds.get('channel_stats', [])
                    if ch_stats:
                        ctx_parts.append(f"  Per-Channel Data:")
                        for cs in ch_stats:
                            ctx_parts.append(f"    {cs['electrode']} (raw: {cs['raw_name']}): "
                                             f"range {cs['amplitude_uV_range']} µV, "
                                             f"std {cs['amplitude_uV_std']} µV")

            pipeline = study_context.get('pipeline_results', {})
            if pipeline and (pipeline.get('band_powers') or pipeline.get('erp_peak')):
                ctx_parts.append(f"\nPipeline Results (status: {pipeline.get('status', 'unknown')}):")
                bp = pipeline.get('band_powers', {})
                if bp:
                    ctx_parts.append(f"  Band Powers: {json.dumps(bp)}")
                erp = pipeline.get('erp_peak', {})
                if erp:
                    ctx_parts.append(f"  ERP Peak: {json.dumps(erp)}")
        ctx_str = "\n".join(ctx_parts) + "\n\n" if ctx_parts else ""

        messages = []
        if conversation_history:
            for msg in conversation_history[-10:]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)

        messages.append({"role": "user", "content": ctx_str + user_prompt})

        try:
            if provider == "kilo":
                content = self._call_kilo(messages, CORTEXIQ_SYSTEM_PROMPT, model_id)
            else:
                content = self._call_anthropic(messages, CORTEXIQ_SYSTEM_PROMPT)

            parsed = _extract_json(content)
            if parsed and isinstance(parsed, dict) and "type" in parsed:
                return parsed
            return {"type": "answer", "message": content}
        except Exception as e:
            return {"type": "error", "message": f"AI Engine error ({model}): {str(e)}"}

    def generate_interpretation(self, results_summary: str, study_context: dict) -> str:
        if not self.client:
            return "AI interpretation unavailable — no API key configured."
        prompt = f"""Given these EEG analysis results, write a 3-sentence plain-English scientific interpretation:
{results_summary}
Study: {study_context.get('conditions', 'EEG recording')}
Respond with ONLY the interpretation text, no JSON."""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Interpretation generation failed: {str(e)}"

    def generate_methods(self, pipeline_steps: list, study_context: dict) -> str:
        if not self.client:
            return "Methods paragraph unavailable — no API key configured."
        steps_text = "\n".join([f"- {s.get('name', 'Step')}: {s.get('tool', 'MNE')}" for s in pipeline_steps])
        prompt = f"""Write a journal-ready methods paragraph for this EEG analysis:
Steps performed:
{steps_text}
Study: {study_context.get('conditions', 'EEG recording')}
Channels: {study_context.get('n_channels', 'N/A')}
Sampling rate: {study_context.get('sfreq', 'N/A')} Hz
Write in past tense, formal scientific style. Respond with ONLY the methods text, no JSON."""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Methods generation failed: {str(e)}"
