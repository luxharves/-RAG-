"""OpenAI-compatible VLM (Qwen3-VL) adapter for Phase 0 smoke testing."""

import base64
from pathlib import Path
from openai import OpenAI
from src.config.settings import settings


class VLMClient:
    """Minimal OpenAI-compatible vision-language model client."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or settings.vlm_base_url
        self.api_key = api_key or settings.vlm_api_key
        self.model = model or settings.vlm_model
        self._client: OpenAI | None = None

    @property
    def is_configured(self) -> bool:
        """Check whether the client has non-placeholder configuration."""
        return (
            self.api_key != "replace-me"
            and self.base_url != "https://api.example.com/v1"
            and self.model != "replace-me"
        )

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def chat_with_image(self, image_path: str | Path, prompt: str) -> tuple[str, dict]:
        """Send a vision chat request with an image.

        Args:
            image_path: Path to a local image file (PNG, JPEG, etc.).
            prompt: Text prompt describing what to ask about the image.

        Returns:
            (response_text, raw_response_dict).
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine MIME type from extension
        ext = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/png")

        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
        raw = {"model": response.model, "usage": response.usage.model_dump() if response.usage else None}
        return content, raw
