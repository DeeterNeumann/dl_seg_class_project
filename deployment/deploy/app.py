import os
import gradio as gr
import numpy as np
from pathlib import Path

from model import load_model
from inference import (
    preprocess_image,
    run_tiled_inference,
    overlay_segmentation,
    SEM_COLORS,
    TER_COLORS,
    SEMANTIC_CLASS_NAMES,
    TER_CLASS_NAMES,
)

DEVICE = "cpu"
MODEL = load_model("weights/run9_best.pt", device=DEVICE).to(DEVICE).eval().float()

EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLE_FILES = sorted(str(p) for p in EXAMPLES_DIR.glob("*.png"))
EXAMPLES = [[p, 0.5] for p in EXAMPLE_FILES]


def build_legend_html(title: str, class_names: dict[int, str], colors: dict[int, tuple[int, int, int] | None]) -> str:
    rows = []
    for k in sorted(class_names.keys()):
        color = colors.get(k)
        name = class_names[k]
        if color is None:
            # still show background, but greyed
            swatch = "<span style='display:inline-block;width:14px;height:14px;background:#ffffff;border:1px solid #999;border-radius:3px;'></span>"
        else:
            r, g, b = color
            swatch = (
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:rgb({r},{g},{b});border:1px solid #333;border-radius:3px;'></span>"
            )
        rows.append(
            f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0;'>"
            f"{swatch}<span style='font-size:14px;'>{k}: {name}</span></div>"
        )

    return f"""
    <div style="padding:12px;border:1px solid #e5e7eb;border-radius:10px;">
      <div style="font-weight:700;font-size:16px;margin-bottom:8px;">{title}</div>
      {''.join(rows)}
    </div>
    """


SEM_LEGEND_HTML = build_legend_html("Semantic classes (cell types)", SEMANTIC_CLASS_NAMES, SEM_COLORS)
TER_LEGEND_HTML = build_legend_html("Ternary classes", TER_CLASS_NAMES, TER_COLORS)


def predict(image, alpha):
    if image is None:
        return None, None

    img_np = preprocess_image(image)  # float32 normalized
    img_rgb = np.array(image.convert("RGB"), dtype=np.uint8)

    sem_pred, ter_pred = run_tiled_inference(MODEL, img_np, device=DEVICE)

    sem_overlay = overlay_segmentation(img_rgb, sem_pred, SEM_COLORS, alpha=alpha)
    ter_overlay = overlay_segmentation(img_rgb, ter_pred, TER_COLORS, alpha=alpha)

    return sem_overlay, ter_overlay


with gr.Blocks(title="MoNuSAC Nucleus Segmentation") as demo:
    gr.Markdown("# 🔬 MoNuSAC Nucleus Segmentation")
    gr.Markdown("Upload a histopathology image to segment nuclei into 4 cell types.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload H&E Image")
            alpha_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Overlay Opacity")
            gr.Markdown("### Try an example")
            gr.Examples(
                examples=EXAMPLES,
                inputs=[input_image, alpha_slider],
                outputs=[sem_output, ter_output],
                fn=predict,
                cache_examples=True,
            )
            run_btn = gr.Button("Run Segmentation", variant="primary")

            gr.Markdown("## Prediction Key")
            gr.HTML(SEM_LEGEND_HTML)
            gr.HTML("<div style='height:10px;'></div>")
            gr.HTML(TER_LEGEND_HTML)

        with gr.Column(scale=2):
            sem_output = gr.Image(label="Semantic Segmentation (cell types)")
            ter_output = gr.Image(label="Ternary Map (inside / boundary / background)")

    run_btn.click(fn=predict, inputs=[input_image, alpha_slider], outputs=[sem_output, ter_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))