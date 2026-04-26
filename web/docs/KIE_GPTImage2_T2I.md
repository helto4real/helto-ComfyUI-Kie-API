# KIE GPT Image 2 (Text-to-Image)

Generate an image from a text prompt using GPT Image 2.

## Inputs
- **prompt** (STRING, required): Text prompt (up to 20000 characters).
- **aspect_ratio** (COMBO, required): `auto`, `1:1`, `9:16`, `16:9`, `4:3`, `3:4`
- **resolution** (COMBO, required): `1K`, `2K`, `4K` (4K not compatible with 1:1 aspect ratio)
- **log** (BOOLEAN, optional): Enable console logging.

## Outputs
- **IMAGE**: ComfyUI image tensor (BHWC, float32, 0–1)

## Helper behavior
- Validates prompt length.
- Creates task with GPT Image 2 T2I model.
- Polls the task until completion, then downloads and decodes the output image.
- Validates that 4K resolution is not used with 1:1 aspect ratio.