# WK3 Level-1.5 Diffusion / Detail Restoration Prototype

This package adds a **pretrained restoration prototype** rather than training a video diffusion model from scratch.

## Files

- `app_motion_ms_poster_module_ui_v6_level15_diffusion.py`  
  Full Gradio replacement with poster-aligned UI, LPIPS checkbox, source selection, comparison video, and Level-1.5 diffusion restoration controls.

- `poster_level15_restore_video.py`  
  Standalone script to restore a rendered video using a local Diffusers img2img model, or `unsharp_preview` for pipeline debugging.

- `make_four_way_comparison.py`  
  Standalone script to create a 2x2 qualitative comparison video.

## Important status wording

- `Level-1.5 diffusion/detail restoration prototype` becomes active only when the checkbox is enabled and a local pretrained Diffusers model is actually executed.
- `unsharp_preview` is not diffusion. It is only for checking the output path and UI wiring.
- `Canonical Gaussian fusion` and `camera-aligned held-out rendering` are still separate future modules.
