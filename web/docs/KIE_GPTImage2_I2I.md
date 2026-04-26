# KIE GPT Image 2 (Image-to-Image)

Edit an image using GPT Image 2.

## Inputs
- **images** (IMAGE, required): Input image batch. 1–16 images supported; all are uploaded.
- **prompt** (STRING, required): Text prompt (up to 20000 characters).
- **aspect_ratio** (COMBO, required): `auto`, `1:1`, `9:16`, `16:9`, `4:3`, `3:4`
- **resolution** (COMBO, required): `1K`, `2K`, `4K` (4K not compatible with 1:1 aspect ratio)
- **log** (BOOLEAN, optional): Enable console logging.

## Outputs
- **IMAGE**: ComfyUI image tensor (BHWC, float32, 0–1)

## Helper behavior
- Validates prompt length and image tensor shape.
- Uploads up to 16 images and passes their URLs in `input_urls`.
- Polls the task until completion, then downloads and decodes the output image.
- Validates that 4K resolution is not used with 1:1 aspect ratio.