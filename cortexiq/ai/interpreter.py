"""CortexIQ AI Interpreter — supports Anthropic Claude and Kilo Gateway."""
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
    "claude": {"provider": "anthropic", "model": "claude-opus-4-6"},
    "kilo-claude": {"provider": "kilo", "model": "anthropic/claude-sonnet-4.6"},
    "kilo-gpt4o": {"provider": "kilo", "model": "openai/gpt-4o"},
    "kilo-gemini": {"provider": "kilo", "model": "google/gemini-2.5-pro"},
}


def _extract_json(text: str) -> dict | None:
    """Multi-pass JSON extraction: handles markdown fences, extra text, and common formatting issues.
    Prioritizes pipeline-type objects when multiple JSON objects are present."""
    if not text:
        return None
    cleaned = text.strip()

    # Pass 1: Direct JSON
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Pass 2: Markdown fences (even if they have text outside)
    fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
    if fenced:
        try:
            candidate = fenced.group(1).strip()
            return json.loads(candidate, strict=False)
        except Exception:
            pass

    # Pass 3: Collect ALL valid JSON objects and pick the best one
    # This handles cases where Claude outputs a preamble {"type":"answer"} before the real JSON
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
                candidate = cleaned[start:i + 1]
                try:
                    obj = json.loads(candidate, strict=False)
                    if isinstance(obj, dict):
                        all_jsons.append(obj)
                except Exception:
                    pass
                start = -1

    if all_jsons:
        # Prefer pipeline > interpretation > clarification > any with type > first
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
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.kilo_key = os.getenv("KILO_API_KEY")
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

    def _call_anthropic(self, messages: list, system: str, model: str = "claude-opus-4-6", max_tokens: int = 8000) -> str:
        """Call Anthropic API directly with retry on overload."""
        if not self.client:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise RuntimeError("Anthropic API Key not set.")
            self.client = Anthropic(api_key=self.api_key)

        import time
        last_error = None
        for attempt in range(4):  # Up to 4 attempts
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
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s
                    print(f"[ANTHROPIC] Overloaded, retry {attempt+1}/4 in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                raise  # Non-overload errors propagate immediately
        raise last_error

    def _call_kilo(self, messages: list, system: str, model: str, max_tokens: int = 4000) -> str:
        """Call Kilo Gateway (OpenAI-compatible) API."""
        kilo_key = self.kilo_key or os.getenv("KILO_API_KEY")
        if not kilo_key:
            raise RuntimeError("Kilo API Key not set.")

        # Gemini reasoning models need extra tokens for internal chain-of-thought
        effective_max = max_tokens
        if "gemini" in model:
            effective_max = max(max_tokens, 8192)

        # Prepend system message
        full_messages = [{"role": "system", "content": system}] + messages

        headers = {
            "Authorization": f"Bearer {kilo_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": full_messages,
            "max_tokens": effective_max,
        }
        resp = requests.post(KILO_BASE_URL, headers=headers, json=payload, timeout=120)

        if resp.status_code != 200:
            try:
                err_body = resp.json()
                err_msg = err_body.get("error", {}).get("message", resp.text)
            except Exception:
                err_msg = resp.text
            logger.error(f"Kilo API error {resp.status_code} (model={model}): {err_msg}")
            raise RuntimeError(f"Kilo API error ({resp.status_code}): {err_msg}")

        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content")

        # Some reasoning models (e.g. Gemini) may return content=null with a reasoning field
        if not content:
            reasoning = msg.get("reasoning")
            if reasoning:
                logger.warning(f"Kilo model {model} returned null content but has reasoning; using reasoning as fallback")
                return reasoning.strip()
            raise RuntimeError(f"Kilo API returned empty content for model={model}")
        return content.strip()

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
                content = self._call_anthropic(messages, CORTEXIQ_SYSTEM_PROMPT, model=model_id)

            # DEBUG: Log raw AI response
            print(f"[INTERP DEBUG] Raw response (first 800): {content[:800]}", flush=True)

            parsed = _extract_json(content)
            if parsed and isinstance(parsed, dict) and "type" in parsed:
                result = self._post_process(parsed, user_prompt)
                print(f"[INTERP DEBUG] Parsed type={result.get('type')}, steps={len(result.get('pipeline_steps', []))}", flush=True)
                return result

            # JSON extraction failed — check if user wanted analysis
            print(f"[INTERP DEBUG] JSON extraction failed or no 'type' field.", flush=True)
            if self._is_analysis_request(user_prompt):
                print(f"[INTERP DEBUG] Detected analysis request, generating pipeline from text...", flush=True)
                steps = self._extract_steps_from_text(content)
                if not steps:
                    print(f"[INTERP DEBUG] Text extraction found no steps, using default pipeline", flush=True)
                    steps = self._generate_default_pipeline()
                if steps:
                    print(f"[INTERP DEBUG] Generated pipeline with {len(steps)} steps from non-JSON response", flush=True)
                    return {
                        "type": "pipeline",
                        "understanding": content[:200] if content else "",
                        "message": "I've set up a processing pipeline for you. Review the steps and click Run when ready.",
                        "pipeline_steps": steps,
                        "suggestions": [],
                        "needs_clarification": False,
                    }
            return {"type": "answer", "message": content}
        except Exception as e:
            err_msg = str(e)
            print(f"[INTERP DEBUG] API error: {err_msg[:200]}", flush=True)
            # If API failed but user asked for analysis, give them a default pipeline anyway
            if self._is_analysis_request(user_prompt):
                steps = self._generate_default_pipeline()
                if steps:
                    print(f"[INTERP DEBUG] API down, returning default pipeline with {len(steps)} steps", flush=True)
                    return {
                        "type": "pipeline",
                        "understanding": f"(AI engine temporarily unavailable) Generated standard preprocessing pipeline.",
                        "message": "AI engine is temporarily unavailable, but I've set up a standard EEG preprocessing pipeline for you. Review the steps and click Run when ready.",
                        "pipeline_steps": steps,
                        "suggestions": [],
                        "needs_clarification": False,
                    }
            return {"type": "error", "message": f"AI Engine error ({model}): {err_msg}"}

    def _post_process(self, response: dict, user_prompt: str) -> dict:
        """Post-process AI response: detect when answer-type responses should be pipelines."""
        if response.get("type") == "pipeline":
            # Ensure pipeline_steps exists and is a list
            if not isinstance(response.get("pipeline_steps"), list):
                response["pipeline_steps"] = []
            # Ensure message field exists for frontend display
            if "message" not in response:
                response["message"] = response.get("understanding", "Pipeline generated.")
            print(f"[POST-PROCESS] Got pipeline directly, steps={len(response['pipeline_steps'])}", flush=True)
            return response

        # If type is "answer" but the user asked for analysis, convert to pipeline
        if response.get("type") == "answer" and self._is_analysis_request(user_prompt):
            print(f"[POST-PROCESS] AI returned 'answer' for analysis request, extracting steps...", flush=True)
            steps = self._extract_steps_from_text(response.get("message", ""))
            if steps:
                print(f"[POST-PROCESS] Extracted {len(steps)} steps from text", flush=True)
                return {
                    "type": "pipeline",
                    "understanding": response.get("message", "")[:200],
                    "message": "I've generated a processing pipeline based on your request.",
                    "pipeline_steps": steps,
                    "suggestions": [],
                    "needs_clarification": False,
                }
            # Fallback: generate a default EEG pipeline even if text extraction failed
            print(f"[POST-PROCESS] Text extraction failed, using default pipeline", flush=True)
            steps = self._generate_default_pipeline()
            if steps:
                return {
                    "type": "pipeline",
                    "understanding": response.get("message", "")[:200],
                    "message": "I've set up a standard EEG preprocessing pipeline for you. Review the steps and click Run when ready.",
                    "pipeline_steps": steps,
                    "suggestions": [],
                    "needs_clarification": False,
                }

        print(f"[POST-PROCESS] No conversion applied, returning type={response.get('type')}", flush=True)
        return response

    def _generate_default_pipeline(self) -> list:
        """Generate a standard default EEG preprocessing pipeline."""
        defaults = [
            ("Band-pass Filtering", "mne.io.Raw.filter",
             {"l_freq": 0.1, "h_freq": 100.0, "notch": 50.0}),
            ("Bad Channel Detection", "mne.preprocessing.find_bad_channels",
             {"z_threshold": 3.0}),
            ("ICA Artifact Removal", "mne.preprocessing.ICA",
             {"n_components": 15}),
            ("Epoching", "mne.make_fixed_length_epochs",
             {"epoch_duration": 2.0}),
            ("PSD Computation", "mne.io.Raw.compute_psd",
             {"fmin": 1.0, "fmax": 45.0}),
        ]
        steps = []
        for i, (name, tool, params) in enumerate(defaults, 1):
            steps.append({
                "step_id": i,
                "name": name,
                "explanation": f"{name} applied to the EEG data.",
                "tool": tool,
                "parameters": params,
                "python_code": EEGPipeline.get_step_code(name, tool, params),
                "what_to_expect": f"After this step, the data will be processed.",
                "can_pause_here": True,
            })
        return steps

    def _is_analysis_request(self, prompt: str) -> bool:
        """Detect if a user prompt is requesting data analysis/processing."""
        prompt_lower = prompt.lower()
        analysis_keywords = [
            "analy", "process", "filter", "clean", "artifact", "ica", "epoch",
            "psd", "spectral", "power", "band", "erp", "evoked", "pipeline",
            "run", "execute", "perform", "remove", "detect", "compute",
            "fourier", "frequency", "time-frequency", "connectivity",
            "reference", "re-reference", "baseline", "segment",
            "denoise", "preprocessing", "pre-process", "what should",
            "help me", "how do i", "can you", "set up", "create",
        ]
        return any(kw in prompt_lower for kw in analysis_keywords)

    def _extract_steps_from_text(self, text: str) -> list:
        """Try to extract pipeline steps from a text description."""
        if not text:
            return []

        # Look for numbered steps or step-like patterns
        steps = []
        lines = text.split('\n')
        step_id = 0

        # Common EEG processing step mappings
        step_mappings = [
            (["filter", "bandpass", "band-pass", "highpass", "lowpass", "notch"],
             {"name": "Band-pass Filtering", "tool": "mne.io.Raw.filter",
              "parameters": {"l_freq": 0.1, "h_freq": 100.0, "notch": 50.0}}),
            (["bad channel", "ransac", "interpolat", "bad ch"],
             {"name": "Bad Channel Detection", "tool": "mne.preprocessing.find_bad_channels",
              "parameters": {"z_threshold": 3.0}}),
            (["ica", "independent component", "artifact removal", "artifact rejection"],
             {"name": "ICA Artifact Removal", "tool": "mne.preprocessing.ICA",
              "parameters": {"n_components": 15}}),
            (["epoch", "segment", "time-lock"],
             {"name": "Epoching", "tool": "mne.make_fixed_length_epochs",
              "parameters": {"epoch_duration": 2.0}}),
            (["psd", "power spectral", "welch", "spectral density", "band power"],
             {"name": "PSD Computation", "tool": "mne.io.Raw.compute_psd",
              "parameters": {"fmin": 1.0, "fmax": 45.0}}),
            (["erp", "evoked", "event-related", "average"],
             {"name": "ERP Analysis", "tool": "mne.Epochs.average",
              "parameters": {}}),
            (["reference", "re-reference", "reref"],
             {"name": "Re-referencing", "tool": "mne.io.Raw.set_eeg_reference",
              "parameters": {"ref": "average"}}),
        ]

        text_lower = text.lower()
        seen_steps = set()
        for keywords, step_template in step_mappings:
            for kw in keywords:
                if kw in text_lower and step_template["name"] not in seen_steps:
                    seen_steps.add(step_template["name"])
                    step_id += 1
                    steps.append({
                        "step_id": step_id,
                        "name": step_template["name"],
                        "explanation": f"{step_template['name']} applied to the EEG data.",
                        "tool": step_template["tool"],
                        "parameters": step_template["parameters"],
                        "python_code": EEGPipeline.get_step_code(
                            step_template["name"], step_template["tool"], step_template["parameters"]
                        ),
                        "what_to_expect": f"After this step, the data will be {step_template['name'].lower()}d.",
                        "can_pause_here": True,
                    })
                    break

        return steps

    def generate_interpretation(self, results_summary: str, study_context: dict) -> str:
        if not self.client:
            return "AI interpretation unavailable — no API key configured."
        prompt = f"""Given these EEG analysis results, write a 3-sentence plain-English scientific interpretation:
{results_summary}
Study: {study_context.get('conditions', 'EEG recording')}
Respond with ONLY the interpretation text, no JSON."""
        try:
            model_id = MODEL_MAP["claude"]["model"]
            return self._call_anthropic(
                messages=[{"role": "user", "content": prompt}],
                system="You are a neuroscience expert. Respond with plain text only, no JSON.",
                model=model_id,
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
            model_id = MODEL_MAP["claude"]["model"]
            return self._call_anthropic(
                messages=[{"role": "user", "content": prompt}],
                system="You are a scientific writer. Respond with plain text only, no JSON.",
                model=model_id,
                max_tokens=500,
            )
        except Exception as e:
            return f"Methods generation failed: {str(e)}"
