"""WAN 2.7 Image image-to-image helper."""

import json
import time
from typing import Any

import torch

from .auth import _load_api_key
from .credits import _log_remaining_credits
from .http import TransientKieError, requests
from .jobs import _poll_task_until_complete
from .log import _log
from .results import _extract_result_urls
from .upload import _image_tensor_to_png_bytes, _truncate_url, _upload_image
from .images import _download_images_as_batch
from .validation import _validate_prompt

CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
MODEL_OPTIONS = ["wan/2-7-image", "wan/2-7-image-pro"]
ASPECT_RATIO_OPTIONS = ["1:1", "16:9", "4:3", "21:9", "3:4", "9:16", "8:1", "1:8"]
RESOLUTION_OPTIONS = ["1K", "2K", "4K"]
PROMPT_MAX_LENGTH = 5000
MAX_INPUT_IMAGES = 9
MAX_OUTPUT_STANDARD = 4
MAX_OUTPUT_SEQUENTIAL = 12


def _validate_images(images: torch.Tensor | None) -> torch.Tensor | None:
    if images is None:
        return None
    if not isinstance(images, torch.Tensor):
        raise RuntimeError("images input must be a tensor batch.")
    if images.dim() != 4 or images.shape[-1] != 3:
        raise RuntimeError("images input must have shape [B, H, W, 3].")
    if images.shape[0] < 1:
        raise RuntimeError("images input batch is empty.")
    if images.shape[0] > MAX_INPUT_IMAGES:
        raise RuntimeError(f"images input batch exceeds maximum of {MAX_INPUT_IMAGES}.")
    return images


def _validate_prompt(prompt: str) -> None:
    if not isinstance(prompt, str):
        raise RuntimeError("prompt must be a string.")
    if not prompt.strip():
        raise RuntimeError("prompt is required.")
    if len(prompt) > PROMPT_MAX_LENGTH:
        raise RuntimeError(
            f"Prompt exceeds maximum length of {PROMPT_MAX_LENGTH} characters."
        )


def _validate_n(n: int, enable_sequential: bool) -> None:
    max_n = MAX_OUTPUT_SEQUENTIAL if enable_sequential else MAX_OUTPUT_STANDARD
    if n < 1 or n > max_n:
        raise RuntimeError(
            f"n must be between 1 and {max_n} when enable_sequential={enable_sequential}."
        )


def _validate_thinking_mode(
    thinking_mode: bool, enable_sequential: bool, has_input_images: bool
) -> None:
    if thinking_mode and (enable_sequential or has_input_images):
        raise RuntimeError(
            "thinking_mode is only available when enable_sequential=false and no input images."
        )


def _validate_resolution_4k(
    resolution: str, model: str, has_input_images: bool
) -> None:
    if resolution == "4K" and model == "wan/2-7-image-pro" and has_input_images:
        raise RuntimeError(
            "4K resolution is only available for text-to-image in Pro mode."
        )


def _create_wan_task(api_key: str, payload: dict[str, Any]) -> tuple[str, str]:
    try:
        response = requests.post(
            CREATE_TASK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to call createTask endpoint: {exc}") from exc

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientKieError(
            f"createTask returned HTTP {response.status_code}: {response.text}",
            status_code=response.status_code,
        )

    try:
        payload_json: Any = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError("createTask endpoint did not return valid JSON.") from exc

    if payload_json.get("code") != 200:
        message = payload_json.get("message") or payload_json.get("msg")
        raise RuntimeError(
            f"createTask endpoint returned error code {payload_json.get('code')}: {message}"
        )

    task_id = (payload_json.get("data") or {}).get("taskId")
    if not task_id:
        raise RuntimeError("createTask endpoint did not return a taskId.")

    return task_id, response.text


def run_wan27_image(
    model: str,
    prompt: str,
    images: torch.Tensor | None,
    aspect_ratio: str,
    resolution: str,
    n: int,
    enable_sequential: bool,
    thinking_mode: bool,
    watermark: bool,
    seed: int,
    poll_interval_s: float = 10.0,
    timeout_s: int = 300,
    log: bool = True,
) -> tuple[torch.Tensor, str]:
    if model not in MODEL_OPTIONS:
        raise RuntimeError("Invalid model. Use the pinned enum options.")
    if aspect_ratio not in ASPECT_RATIO_OPTIONS:
        raise RuntimeError("Invalid aspect_ratio. Use the pinned enum options.")
    if resolution not in RESOLUTION_OPTIONS:
        raise RuntimeError("Invalid resolution. Use the pinned enum options.")

    _validate_prompt(prompt)
    images = _validate_images(images)
    _validate_n(n, enable_sequential)
    _validate_thinking_mode(thinking_mode, enable_sequential, images is not None)
    _validate_resolution_4k(resolution, model, images is not None)

    api_key = _load_api_key()

    image_urls: list[str] = []
    if images is not None and images.shape[0] > 0:
        _log(log, f"Uploading {images.shape[0]} input image(s) for WAN 2.7 Image...")
        for idx in range(images.shape[0]):
            png_bytes = _image_tensor_to_png_bytes(images[idx])
            url = _upload_image(api_key, png_bytes)
            image_urls.append(url)
            _log(log, f"Image {idx + 1} upload success: {_truncate_url(url)}")

    input_payload: dict[str, Any] = {
        "prompt": prompt,
        "n": n,
        "resolution": resolution,
        "watermark": watermark,
        "seed": seed,
        "enable_sequential": enable_sequential,
    }

    if image_urls:
        input_payload["input_urls"] = image_urls

    if aspect_ratio and not image_urls:
        input_payload["aspect_ratio"] = aspect_ratio

    if thinking_mode and not enable_sequential and not image_urls:
        input_payload["thinking_mode"] = True

    payload = {
        "model": model,
        "input": input_payload,
    }

    _log(log, f"Creating WAN 2.7 Image task (model: {model})...")
    start_time = time.time()
    task_id, create_response_text = _create_wan_task(api_key, payload)
    _log(
        log,
        f"createTask response (elapsed={time.time() - start_time:.1f}s): {create_response_text}",
    )
    _log(log, f"Task created with ID {task_id}. Polling for completion...")

    record_data = _poll_task_until_complete(
        api_key,
        task_id,
        poll_interval_s,
        timeout_s,
        log,
        start_time,
    )

    result_urls = _extract_result_urls(record_data)
    _log(log, f"Result URLs: {result_urls}")
    _log(log, f"Downloading {len(result_urls)} result image(s)...")

    image_batch = _download_images_as_batch(result_urls)
    _log(log, "Images downloaded and decoded.")

    _log_remaining_credits(log, record_data, api_key, _log)
    return image_batch, task_id
