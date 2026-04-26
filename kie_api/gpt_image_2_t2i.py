"""GPT Image 2 Text-to-Image helper."""

import time

import torch

from .auth import _load_api_key
from .credits import _log_remaining_credits
from .images import _download_image, _image_bytes_to_tensor
from .jobs import _create_task, _poll_task_until_complete
from .log import _log
from .results import _extract_result_urls
from .validation import _validate_prompt


MODEL_NAME = "gpt-image-2-text-to-image"
ASPECT_RATIO_OPTIONS = ["auto", "1:1", "9:16", "16:9", "4:3", "3:4"]
RESOLUTION_OPTIONS = ["1K", "2K", "4K"]
PROMPT_MAX_LENGTH = 20000


def _validate_options(aspect_ratio: str, resolution: str) -> None:
    if aspect_ratio not in ASPECT_RATIO_OPTIONS:
        raise RuntimeError("Invalid aspect_ratio. Use the pinned enum options.")
    if resolution not in RESOLUTION_OPTIONS:
        raise RuntimeError("Invalid resolution. Use the pinned enum options.")


def _validate_resolution_aspect_combo(resolution: str, aspect_ratio: str) -> None:
    if resolution == "4K" and aspect_ratio == "1:1":
        raise RuntimeError("4K resolution is not compatible with 1:1 aspect ratio.")


def run_gpt_image_2_t2i(
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    poll_interval_s: float,
    timeout_s: int,
    log: bool,
) -> torch.Tensor:
    """Run a GPT Image 2 Text-to-Image job.

    Args:
        prompt: Text prompt.
        aspect_ratio: Output aspect ratio (spec-defined enum).
        resolution: Output resolution (1K, 2K, 4K).
        poll_interval_s: Seconds between status polls.
        timeout_s: Maximum seconds to wait for completion.
        log: Enable verbose logging.

    Returns:
        ComfyUI IMAGE tensor (1, H, W, 3) float32 in [0, 1].

    Raises:
        RuntimeError: For validation errors or non-retryable API/task failures.
    """
    _validate_prompt(prompt, max_length=PROMPT_MAX_LENGTH)
    _validate_options(aspect_ratio, resolution)
    _validate_resolution_aspect_combo(resolution, aspect_ratio)

    api_key = _load_api_key()

    payload = {
        "model": MODEL_NAME,
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
        },
    }

    _log(log, "Creating GPT Image 2 T2I task...")
    start_time = time.time()
    task_id, create_response_text = _create_task(api_key, payload)
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
    _log(log, f"Downloading result image from {result_urls[0]}...")

    image_bytes = _download_image(result_urls[0])
    image_tensor = _image_bytes_to_tensor(image_bytes)
    _log(log, "Image downloaded and decoded.")

    _log_remaining_credits(log, record_data, api_key, _log)
    return image_tensor