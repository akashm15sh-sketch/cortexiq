"""CortexIQ AI Interpreter — supports Anthropic Claude (streaming) and Kilo Gateway."""
import os
import re
import json
import requests
from anthropic import Anthropic
from .system_prompt import CORTEXIQ_SYSTEM_PROMPT
from ..eeg.pipeline import EEGPipeline


import logging
logger = logging.getLogger(__name__)

KILO_BASE_URL = "https://api.kilo.ai/api/gateway/chat/completions"

# Map friendly model names to provider identifiers
MODEL_MAP = {
    "claude":       {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "kilo-claude":  {"provider": "kilo",      "model": "anthropic/claude-sonnet-4.6"},
    "kilo-gpt4o":   {"provider": "kilo",      "model": "openai/gpt-4o"},
    "kilo-gemini":  {"provider": "kilo",      "model": "google/gemini-2.5-pro"},
}

# Token budgets
# 6000 gives headroom for a full 6-step pipeline with explanations.
# Answers typically use <1000 tokens so this doesn't hurt latency much.
MAX_TOKENS_ANSWER   = 1500
MAX_TOKENS_PIPELINE = 6000
MAX_TOKENS_DEFAULT  = 6000


def _extract_json(text: str) -> dict | None:
    """Multi-pass JSON extraction: markdown fences, bare objects, mixed-text."""
    if not text:
        return None
    cleaned = text.strip()

    # Pass 1: Direct JSON
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Pass 2: Markdown fences
    fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip(), strict=False)
        except Exception:
            pass

    # Pass 3: Collect ALL valid JSON objects; prefer pipeline > interpretation > any-with-type
    all_jsons = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(cleaned[start:i + 1], strict=False)
                    if isinstance(obj, dict):
                        all_jsons.append(obj)
                except Exception:
                    pass
                start = -1

    if all_jsons:
        for obj in all_jsons:
            if obj.get("type") == "pipeline" and "pipeline_steps" in obj:
                return obj
        for obj in all_jsons:
            if obj.get("type") == "pipeline":
                return obj
        for obj in all_jsons:
            if obj.get("type") in ("interpretation", "clarification", "refusal"):
                return obj
        for obj in all_jsons:
            if "type" in obj:
                return obj
        return all_jsons[0]

    logger.error("JSON extraction total failure")
    return None


class CortexIQInterpreter:
    def __init__(self):
        self.api_key  = os.getenv("ANTHROPIC_API_KEY")
        self.kilo_key = os.getenv("KILO_API_KEY")
        self.client   = Anthropic(api_key=self.api_key) if self.api_key else None

    # ── context & message helpers ────────────────────────────────────────────

    def _build_context_string(self, study_context: dict) -> str:
        """Build the plain-text context block prepended to every user message."""
        if not study_context:
            return ""

        parts = ["Study Context:"]
        parts.append(f"  Study Name: {study_context.get('name', 'Untitled')}")
        parts.append(f"  Modality: {study_context.get('modality', 'EEG')}")
        parts.append(f"  Format: {study_context.get('file_format', 'unknown')}")
        parts.append(f"  Total Subjects: {study_context.get('n_subjects', 'unknown')}")
        parts.append(f"  Channels: {study_context.get('n_channels', 'unknown')}")

        ch_names = study_context.get('channel_names', [])
        if ch_names:
            if len(ch_names) > 32:
                parts.append(f"  Channel Names (truncated): {', '.join(ch_names[:32])} ...")
            else:
                parts.append(f"  Channel Names: {', '.join(ch_names)}")

        channel_mapping = study_context.get('channel_mapping', [])
        if channel_mapping:
            mapping_strs = [f"Ch{m['index']}:{m['raw']}→{m['label']}" for m in channel_mapping[:32]]
            parts.append(f"  Channel Renames ({len(channel_mapping)}): {', '.join(mapping_strs)}"
                         + (" ..." if len(channel_mapping) > 32 else ""))

        parts.append(f"  Sampling Rate: {study_context.get('sfreq', 'unknown')} Hz")
        parts.append(f"  Duration: {study_context.get('duration_sec', 'unknown')} sec")
        parts.append(f"  Total Recording Duration: {study_context.get('total_duration_sec', 'unknown')} sec")
        parts.append(f"  Conditions: {study_context.get('conditions', 'not specified')}")
        parts.append(f"  Reference: {study_context.get('reference', 'not specified')}")

        montage = study_context.get('montage', [])
        mapped_electrodes = [m for m in montage if m]
        if mapped_electrodes:
            if len(mapped_electrodes) > 32:
                parts.append(f"  Electrode Montage ({len(mapped_electrodes)} mapped): "
                             f"{', '.join(mapped_electrodes[:32])} ...")
            else:
                parts.append(f"  Electrode Montage ({len(mapped_electrodes)} mapped): "
                             f"{', '.join(mapped_electrodes)}")

        if study_context.get('events'):
            ev = study_context['events']
            parts.append(f"  Events/Stimulus File: {ev.get('name', 'unknown')} "
                         f"({ev.get('size', 0)} bytes)")
            content = ev.get('content', '')
            if content and isinstance(content, str):
                preview = content[:800].strip()
                parts.append(f"  Event File Content (preview):\n{preview}"
                             + ("\n  [truncated]" if len(content) > 800 else ""))

        if study_context.get('notes'):
            parts.append(f"  Notes: {study_context['notes']}")

        subjects = study_context.get('subjects', [])
        if subjects:
            parts.append("\nPer-Subject Details:")
            for i, subj in enumerate(subjects):
                parts.append(f"  Subject {i+1} ({subj.get('name', 'unknown')}): "
                             f"{subj.get('n_channels', '?')} channels, "
                             f"{subj.get('sfreq', '?')} Hz, "
                             f"{subj.get('duration_sec', '?')} sec, "
                             f"format={subj.get('format', '?')}")

        data_stats = study_context.get('data_stats', [])
        if data_stats:
            parts.append("\nData Statistics:")
            for i, ds in enumerate(data_stats):
                parts.append(f"  Subject {i+1} ({ds.get('name', '?')}): "
                             f"{ds.get('n_samples', '?')} samples, "
                             f"amplitude range: {ds.get('amplitude_uV_range', '?')} µV, "
                             f"std: {ds.get('amplitude_uV_std', '?')} µV")
                ch_stats = ds.get('channel_stats', [])
                if ch_stats:
                    parts.append("  Per-Channel Data:")
                    for cs in ch_stats:
                        parts.append(f"    {cs['electrode']} (raw: {cs['raw_name']}): "
                                     f"range {cs['amplitude_uV_range']} µV, "
                                     f"std {cs['amplitude_uV_std']} µV")

        pipeline = study_context.get('pipeline_results', {})
        if pipeline and (pipeline.get('band_powers') or pipeline.get('erp_peak')):
            parts.append(f"\nPipeline Results (status: {pipeline.get('status', 'unknown')}):")
            bp = pipeline.get('band_powers', {})
            if bp:
                parts.append(f"  Band Powers: {json.dumps(bp)}")
            erp = pipeline.get('erp_peak', {})
            if erp:
                parts.append(f"  ERP Peak: {json.dumps(erp)}")

        return "\n".join(parts) + "\n\n"

    def _build_messages(self, conversation_history: list, ctx_str: str, user_prompt: str) -> list:
        messages = []
        if conversation_history:
            for m in conversation_history[-10:]:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    messages.append(m)
        messages.append({"role": "user", "content": ctx_str + user_prompt})
        return messages

    # ── Anthropic calls ──────────────────────────────────────────────────────

    def _call_anthropic(self, messages: list, system: str,
                        model: str = "claude-sonnet-4-6",
                        max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        """Blocking Anthropic call with retry on overload."""
        if not self.client:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise RuntimeError("Anthropic API Key not set.")
            self.client = Anthropic(api_key=self.api_key)

        import time
        last_error = None
        for attempt in range(4):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                return response.content[0].text.strip()
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "529" in err_str or "overloaded" in err_str.lower():
                    wait = 2 ** attempt
                    print(f"[ANTHROPIC] Overloaded, retry {attempt+1}/4 in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise
        raise last_error

    def _stream_anthropic(self, messages: list, system: str,
                          model: str = "claude-sonnet-4-6",
                          max_tokens: int = MAX_TOKENS_DEFAULT):
        """Generator: yields text chunks via Anthropic streaming API."""
        if not self.client:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise RuntimeError("Anthropic API Key not set.")
            self.client = Anthropic(api_key=self.api_key)

        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk

    # ── Kilo call ────────────────────────────────────────────────────────────

    def _call_kilo(self, messages: list, system: str,
                   model: str, max_tokens: int = MAX_TOKENS_DEFAULT) -> str:
        kilo_key = self.kilo_key or os.getenv("KILO_API_KEY")
        if not kilo_key:
            raise RuntimeError("Kilo API Key not set.")

        effective_max = max(max_tokens, 8192) if "gemini" in model else max_tokens
        full_messages = [{"role": "system", "content": system}] + messages

        resp = requests.post(
            KILO_BASE_URL,
            headers={"Authorization": f"Bearer {kilo_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": full_messages, "max_tokens": effective_max},
            timeout=120,
        )
        if resp.status_code != 200:
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err_msg = resp.text
            raise RuntimeError(f"Kilo API error ({resp.status_code}): {err_msg}")

        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content")
        if not content:
            reasoning = msg.get("reasoning")
            if reasoning:
                return reasoning.strip()
            raise RuntimeError(f"Kilo API returned empty content for model={model}")
        return content.strip()

    # ── Public: blocking interpret ───────────────────────────────────────────

    def interpret(self, user_prompt: str, study_context: dict,
                  conversation_history: list = None,
                  model: str = "claude") -> dict:
        model_config = MODEL_MAP.get(model, MODEL_MAP["claude"])
        provider  = model_config["provider"]
        model_id  = model_config["model"]

        ctx_str  = self._build_context_string(study_context)
        messages = self._build_messages(conversation_history, ctx_str, user_prompt)
        max_tok  = MAX_TOKENS_DEFAULT

        try:
            if provider == "kilo":
                content = self._call_kilo(messages, CORTEXIQ_SYSTEM_PROMPT, model_id, max_tok)
            else:
                content = self._call_anthropic(messages, CORTEXIQ_SYSTEM_PROMPT,
                                               model=model_id, max_tokens=max_tok)

            print(f"[INTERP DEBUG] Raw response (first 600): {content[:600]}", flush=True)

            parsed = _extract_json(content)
            if parsed and isinstance(parsed, dict) and "type" in parsed:
                result = self._post_process(parsed, user_prompt)
                print(f"[INTERP DEBUG] type={result.get('type')}, steps={len(result.get('pipeline_steps', []))}", flush=True)
                return result

            # Fallback: treat raw text as answer
            print("[INTERP DEBUG] JSON extraction failed — returning raw text as answer.", flush=True)
            return {"type": "answer", "message": content}

        except Exception as e:
            err_msg = str(e)
            print(f"[INTERP DEBUG] API error: {err_msg[:200]}", flush=True)
            return {"type": "error", "message": f"AI Engine error ({model}): {err_msg}"}

    # ── Public: streaming interpret ──────────────────────────────────────────

    def interpret_stream(self, user_prompt: str, study_context: dict,
                         conversation_history: list = None,
                         model: str = "claude"):
        """
        Generator. Yields dicts:
          {'type': 'chunk', 'text': '<partial text>'}   — one per streamed token
          {'type': 'done',  'response': {...}}           — final parsed response
        """
        model_config = MODEL_MAP.get(model, MODEL_MAP["claude"])
        provider  = model_config["provider"]
        model_id  = model_config["model"]

        ctx_str  = self._build_context_string(study_context)
        messages = self._build_messages(conversation_history, ctx_str, user_prompt)
        max_tok  = MAX_TOKENS_DEFAULT

        full_text = ""
        try:
            if provider == "anthropic":
                for chunk in self._stream_anthropic(messages, CORTEXIQ_SYSTEM_PROMPT,
                                                    model=model_id, max_tokens=max_tok):
                    full_text += chunk
                    yield {"type": "chunk", "text": chunk}
            else:
                # Kilo: no streaming — yield whole response as one chunk
                content = self._call_kilo(messages, CORTEXIQ_SYSTEM_PROMPT, model_id, max_tok)
                full_text = content
                yield {"type": "chunk", "text": content}

        except Exception as e:
            yield {"type": "done", "response": {"type": "error", "message": str(e)}}
            return

        # Parse the accumulated response
        print(f"[STREAM] Full response length: {len(full_text)} chars", flush=True)
        print(f"[STREAM] First 200 chars: {full_text[:200]}", flush=True)
        parsed = _extract_json(full_text)
        if parsed and isinstance(parsed, dict) and "type" in parsed:
            response = self._post_process(parsed, user_prompt)
        else:
            print(f"[STREAM] JSON extraction failed — raw text fallback. Last 100: {full_text[-100:]}", flush=True)
            # Try to rescue a pipeline type if text starts with {"type":"pipeline"
            if '"type":"pipeline"' in full_text or '"type": "pipeline"' in full_text:
                print("[STREAM] Detected truncated pipeline — using defaults", flush=True)
                response = {
                    "type": "pipeline",
                    "message": "I've prepared a standard EEG preprocessing pipeline for your data.",
                    "pipeline_steps": self._generate_default_pipeline(),
                }
            else:
                response = {"type": "answer", "message": full_text}

        yield {"type": "done", "response": response}

    # ── Post-processing ──────────────────────────────────────────────────────

    def _post_process(self, response: dict, user_prompt: str) -> dict:
        """Ensure required fields exist. Recover truncated pipeline responses."""
        if response.get("type") == "pipeline":
            if not isinstance(response.get("pipeline_steps"), list):
                # JSON was truncated before steps completed — fall back to defaults
                print("[POST-PROCESS] Pipeline has no steps (likely truncated) — using defaults", flush=True)
                response["pipeline_steps"] = self._generate_default_pipeline()
                response["message"] = (
                    response.get("message") or response.get("understanding") or
                    "I've prepared a standard preprocessing pipeline. You can run it or ask me to customise it."
                )
            elif len(response["pipeline_steps"]) == 0:
                response["pipeline_steps"] = self._generate_default_pipeline()
                response["message"] = response.get("message") or "Standard pipeline ready."
            else:
                if "message" not in response:
                    response["message"] = response.get("understanding", "Pipeline generated.")
            print(f"[POST-PROCESS] Pipeline with {len(response['pipeline_steps'])} steps", flush=True)
            return response

        # For answer/clarification/refusal: just ensure message field exists
        if "message" not in response:
            response["message"] = response.get("clarification_question",
                                   response.get("understanding", ""))
        print(f"[POST-PROCESS] type={response.get('type')}", flush=True)
        return response

    # ── Report helpers ───────────────────────────────────────────────────────

    def generate_interpretation(self, results_summary: str, study_context: dict) -> str:
        if not self.client:
            return "AI interpretation unavailable — no API key configured."
        prompt = f"""Given these EEG analysis results, write a 3-sentence plain-English scientific interpretation:
{results_summary}
Study: {study_context.get('conditions', 'EEG recording')}
Respond with ONLY the interpretation text, no JSON."""
        try:
            return self._call_anthropic(
                messages=[{"role": "user", "content": prompt}],
                system="You are a neuroscience expert. Respond with plain text only, no JSON.",
                model=MODEL_MAP["claude"]["model"],
                max_tokens=500,
            )
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
            return self._call_anthropic(
                messages=[{"role": "user", "content": prompt}],
                system="You are a scientific writer. Respond with plain text only, no JSON.",
                model=MODEL_MAP["claude"]["model"],
                max_tokens=500,
            )
        except Exception as e:
            return f"Methods generation failed: {str(e)}"

    # ── Legacy fallback helpers (kept for report generation) ─────────────────

    def _generate_default_pipeline(self) -> list:
        defaults = [
            ("Band-pass Filtering",     "mne.io.Raw.filter",                    {"l_freq": 0.1, "h_freq": 100.0, "notch": 50.0}),
            ("Bad Channel Detection",   "mne.preprocessing.find_bad_channels",  {"z_threshold": 3.0}),
            ("ICA Artifact Removal",    "mne.preprocessing.ICA",                {"n_components": 15}),
            ("Epoching",                "mne.make_fixed_length_epochs",         {"epoch_duration": 2.0}),
            ("PSD Computation",         "mne.io.Raw.compute_psd",               {"fmin": 1.0, "fmax": 45.0}),
        ]
        return [
            {
                "step_id": i,
                "name": name,
                "why": f"{name} is a standard preprocessing step.",
                "when_to_use": "Included by default in EEG preprocessing.",
                "explanation": f"{name} applied to the EEG data.",
                "tool": tool,
                "parameters": params,
                "python_code": EEGPipeline.get_step_code(name, tool, params),
                "code_explanation": f"Applies {name} using MNE-Python.",
                "what_to_expect": "Data is cleaner after this step.",
                "can_pause_here": True,
            }
            for i, (name, tool, params) in enumerate(defaults, 1)
        ]
