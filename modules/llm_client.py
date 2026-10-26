"""
Gemini Vision LLM Client for VisionTalk AI
Handles multi-modal image understanding, follow-up conversation turns,
RAG context injection, and robust error handling (REST + SDK).
"""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

try:
    import google.generativeai as genai
    GENAI_SDK_AVAILABLE = True
except ImportError:
    GENAI_SDK_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class GeminiVisionClient:
    """Multi-modal Gemini client with dual SDK/REST architecture and resilient error handling."""

    DEFAULT_MODELS = [
        "gemini-3.8-flash",
        "gemini-3.8-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.8-flash"):
        raw_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.api_key = raw_key.strip()
        self.model_name = model_name.strip()
        self._sdk_configured = False
        self._cached_available_models: List[str] = []

        if self.api_key and GENAI_SDK_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self._sdk_configured = True
            except Exception:
                self._sdk_configured = False

    def is_configured(self) -> bool:
        """Returns True if an API key is available."""
        return bool(self.api_key and self.api_key.strip())

    def update_credentials(self, api_key: str, model_name: Optional[str] = None) -> None:
        """Updates runtime API key and model."""
        new_key = api_key.strip() if api_key else ""
        if new_key != self.api_key:
            self._cached_available_models = []
        self.api_key = new_key
        if model_name:
            self.model_name = model_name.strip()
        if self.api_key and GENAI_SDK_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self._sdk_configured = True
            except Exception:
                self._sdk_configured = False

    def list_available_models(self, force_refresh: bool = False) -> Tuple[bool, List[str], str]:
        """Queries the Gemini API for models available to this API key."""
        if not self.is_configured():
            return False, [], "API key is not configured."
        if not REQUESTS_AVAILABLE:
            return False, [], "requests package is not available."

        if not force_refresh and self._cached_available_models:
            return True, self._cached_available_models, "Success"

        key = self.api_key.strip()
        for api_ver in ["v1beta", "v1"]:
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={key}"
            try:
                res = requests.get(url, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    models = []
                    for m in data.get("models", []):
                        methods = m.get("supportedGenerationMethods", [])
                        if "generateContent" in methods:
                            name = m.get("name", "").replace("models/", "")
                            if name:
                                models.append(name)
                    if models:
                        self._cached_available_models = models
                        return True, models, "Success"
                elif res.status_code in (401, 403):
                    return False, [], "Invalid API key or access denied."
                elif res.status_code == 400:
                    return False, [], f"Google API error: {res.text}"
            except Exception as e:
                pass
        return False, [], "Unable to fetch model list. Check API key and internet connectivity."

    @staticmethod
    def _image_to_base64_data(image: Image.Image) -> Tuple[str, str]:
        """Converts PIL Image to base64 string and MIME type."""
        buffered = io.BytesIO()
        fmt = image.format if image.format in ["JPEG", "PNG", "WEBP"] else "JPEG"
        if image.mode in ("RGBA", "P") and fmt == "JPEG":
            image = image.convert("RGB")
        image.save(buffered, format=fmt)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        mime = f"image/{fmt.lower()}"
        return b64_str, mime

    def _call_rest_api(
        self,
        contents: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Direct REST API with dynamic ModelService discovery and detailed Google error reporting."""
        if not REQUESTS_AVAILABLE:
            return False, "Neither google-generativeai SDK nor requests package is available."

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048,
            }
        }
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        headers = {"Content-Type": "application/json"}
        key = self.api_key.strip()
        clean_model = self.model_name.split("/")[-1].strip()

        # Step 1: Dynamically query Google's ModelService to get exact supported models for this key
        ok, live_models, list_err = self.list_available_models()
        candidate_models: List[str] = []

        if ok and live_models:
            # If the currently selected model is supported by Google, try it first
            if clean_model in live_models:
                candidate_models.append(clean_model)
            # Add flash models from live models
            for m in live_models:
                if "flash" in m.lower() and m not in candidate_models:
                    candidate_models.append(m)
            # Add any other generateContent models returned by Google
            for m in live_models:
                if m not in candidate_models:
                    candidate_models.append(m)
        else:
            # Fallback candidate list if ListModels is unavailable
            candidate_models = [clean_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

        # Step 2: Attempt generation with candidate models
        last_error = ""

        for mod in candidate_models:
            clean_mod = mod.split("/")[-1].strip()
            for api_ver in ["v1beta", "v1"]:
                url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{clean_mod}:generateContent?key={key}"
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        candidates_resp = data.get("candidates", [])
                        if candidates_resp:
                            parts = candidates_resp[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts)
                            if text:
                                self.model_name = clean_mod
                                return True, text.strip()
                        return False, "Model returned an empty response. Image content may have triggered safety filters."
                    elif response.status_code == 400:
                        last_error = f"Bad Request (400): {response.text}"
                    elif response.status_code in (401, 403):
                        return False, "Authentication Error: Invalid API key or access denied. Please verify your API key from Google AI Studio."
                    elif response.status_code == 429:
                        return False, "API Quota Exceeded (429): Rate limit exceeded. Please wait a moment before trying again."
                    else:
                        try:
                            err_msg = response.json().get("error", {}).get("message", response.text)
                        except Exception:
                            err_msg = response.text
                        last_error = f"Gemini API ({clean_mod} - HTTP {response.status_code}): {err_msg}"
                except requests.exceptions.Timeout:
                    return False, "Connection Timeout: Gemini API did not respond within 60 seconds."
                except requests.exceptions.ConnectionError:
                    return False, "Network Error: Could not connect to Gemini API. Check your internet connection."
                except Exception as e:
                    last_error = str(e)

        if list_err and "Success" not in list_err:
            return False, f"Could not reach Gemini model. ModelService note: {list_err}. Last error: {last_error}"
        return False, last_error or "Unable to generate response from Gemini API."

    def generate_chat_response(
        self,
        image: Optional[Image.Image],
        user_query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        retrieved_context: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Generates conversational response using image, detected objects, RAG context, and prior turns.
        Returns: (success: bool, response_or_error_message: str)
        """
        if not self.is_configured():
            return False, (
                "⚠️ **Gemini API Key Missing**: Please set the `GEMINI_API_KEY` environment variable "
                "or enter your API key in the sidebar to enable conversational image QA."
            )

        # Build System Instructions
        system_instruction = (
            "You are VisionTalk AI, an expert conversational computer vision assistant. "
            "You analyze images, answer user questions, explain detected objects, and provide factual, "
            "detailed, and helpful insights.\n\n"
            "Guidelines:\n"
            "1. Ground your answers in the uploaded image, object detection facts, and domain knowledge retrieved.\n"
            "2. If domain knowledge (RAG) is provided, cite relevant facts and sources accurately.\n"
            "3. If object detection data is provided, reference the identified objects and confidence scores when relevant.\n"
            "4. Maintain conversational context across follow-up turns.\n"
            "5. If an object is not in the image, clearly state that it is not visible."
        )

        # Build Grounding Context
        grounding_notes = []
        if detected_objects:
            obj_list_str = ", ".join(
                f"{obj.get('label', 'object')} (confidence: {obj.get('confidence', 0.0):.1%})"
                for obj in detected_objects
            )
            grounding_notes.append(f"**YOLOv8 Real Detections:** {obj_list_str}")
        elif detected_objects is not None and len(detected_objects) == 0:
            grounding_notes.append("**YOLOv8 Detection Note:** No objects exceeded the detection confidence threshold.")

        if retrieved_context and retrieved_context.strip():
            grounding_notes.append(f"**Retrieved Knowledge (RAG):**\n{retrieved_context.strip()}")

        context_header = "\n\n".join(grounding_notes)

        # Build message history for REST payload
        contents: List[Dict[str, Any]] = []

        # Include prior conversation turns if available
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role")
                content = turn.get("content", "")
                if role in ("user", "assistant"):
                    gemini_role = "user" if role == "user" else "model"
                    contents.append({
                        "role": gemini_role,
                        "parts": [{"text": content}]
                    })

        # Build current user message parts
        current_parts: List[Dict[str, Any]] = []
        if image is not None:
            try:
                b64_data, mime = self._image_to_base64_data(image)
                current_parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": b64_data
                    }
                })
            except Exception as img_err:
                return False, f"Failed to encode image: {str(img_err)}"

        query_with_context = user_query
        if context_header:
            query_with_context = f"{context_header}\n\n---\n**User Question:** {user_query}"

        current_parts.append({"text": query_with_context})
        contents.append({
            "role": "user",
            "parts": current_parts
        })

        # Try SDK first if configured
        if GENAI_SDK_AVAILABLE and self._sdk_configured:
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                # Prepare contents for SDK
                sdk_contents = []
                if image is not None:
                    sdk_contents.append(image)
                sdk_contents.append(query_with_context)

                response = model.generate_content(sdk_contents)
                if response and response.text:
                    return True, response.text.strip()
            except Exception as sdk_err:
                # If SDK fails, fall back to REST automatically
                pass

        # Call REST API
        return self._call_rest_api(contents, system_instruction=system_instruction)
