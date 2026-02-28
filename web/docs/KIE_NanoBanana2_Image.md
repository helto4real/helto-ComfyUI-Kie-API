# KIE Nano Banana 2 (Image)

Generate images using KIE's Nano Banana 2 model.

This node supports optional reference images, Google web-search grounding,
and the same async task flow as the other KIE image nodes.

---

## Inputs

- **Prompt**  
  Text description of the image to generate.

- **Images (optional)**  
  Up to 14 reference images used for guidance.

- **Google Search**  
  Enable Google web-search grounding for real-time information.

- **Aspect Ratio**  
  Output image aspect ratio.

- **Resolution**  
  `1K`, `2K`, or `4K`.

- **Output Format**  
  `jpg` or `png`.

- **Poll Interval**  
  How often the node checks task status (seconds).

- **Timeout**  
  Maximum wait time before failing (seconds).

- **Log**  
  Enable progress output in the console.

---

## Outputs

- **IMAGE**  
  ComfyUI image tensor (BHWC, float32, range 0-1).

---

## Notes

- Images are uploaded via KIE upload endpoint and sent as URL list in `image_input`.
- If more than 14 images are provided, only the first 14 are used.
- Credits are consumed per request.
