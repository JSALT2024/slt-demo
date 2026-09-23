import gradio as gr
import os
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from handle_gradio import (
    process_video,
    handle_file_upload,
    handle_remove_video,
    handle_select_example,
    open_modal,
    open_modal_for_recording,
    close_modal_and_save,
    cancel_modal,
    flip_video_horizontal,
)

# Check and download the 1GB pre-trained model weights if not cached
print("Checking for large model file...")
local_model_path = os.path.join("Uni_Sign", "unisign_model", "best_checkpoint-wlasl.pth")
if os.path.exists(local_model_path):
    model_path = local_model_path
else:
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="plice13/sign-language-weights", 
        filename="best_checkpoint-wlasl.pth"                  
    )
print(f"File successfully loaded at: {model_path}")
os.environ["UNISIGN_WEIGHTS"] = model_path
# ====================

# Prepare base64-encoded logos for header badges from the 'logo' directory
logo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo")

fav_logo_path = os.path.join(logo_dir, "fav_logo.png")
fav_logo_src = ""
if os.path.exists(fav_logo_path):
    with open(fav_logo_path, "rb") as f:
        fav_b64 = base64.b64encode(f.read()).decode("utf-8")
        fav_logo_src = f"data:image/png;base64,{fav_b64}"

zcu_logo_path = os.path.join(logo_dir, "zcu_logo.png")
zcu_logo_src = ""
if os.path.exists(zcu_logo_path):
    with open(zcu_logo_path, "rb") as f:
        zcu_b64 = base64.b64encode(f.read()).decode("utf-8")
        zcu_logo_src = f"data:image/png;base64,{zcu_b64}"

# Paths to the 3 example videos from examples/wlasl_new
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "displayed")

example_videos = [
    os.path.join(example_dir, "book.mp4"),
    os.path.join(example_dir, "deaf.mp4"),
    os.path.join(example_dir, "help.mp4"),
    os.path.join(example_dir, "fine.mp4"),
    os.path.join(example_dir, "woman.mp4"),
    os.path.join(example_dir, "no.mp4"),
]

# Load CSS from external style.css file
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    custom_css = f.read()

# Information text hardcoded from DELETE/slt_demot.md
INFO_MARKDOWN = """# Sign Language Translation Demo

This interactive demo showcases **automatic sign language translation from video to text**.

The current version demonstrates the processing of **American Sign Language (ASL)** using modern computer vision and deep learning methods.

## How to use the demo

1. **Upload a video** or select one of the provided examples.
2. Click **Submit** to process the video.
3. The predicted text will be displayed as the output.

For best results, the signer should be clearly visible, including the upper body, hands, and face.

> **Note:** This is a research demonstrator. Predictions may not always be accurate, especially for videos that differ significantly from the data used during training.

## How does it work?

The system consists of two main stages: **pose preprocessing** and **sign language translation**.

### Pose preprocessing

The input video is first converted into a structured pose representation. Keypoints describing the signer's **body, hands, and face** are extracted and normalized before being passed to the translation model.

Body keypoints are normalized globally, while hand and facial keypoints are normalized locally to preserve detailed information about their shape and movement.

More information about the preprocessing pipeline is available in the [PoseEstimation repository](https://github.com/JSALT2024/PoseEstimation).

### Sign language translation

The extracted pose sequence is processed using a model based on **Uni-Sign**, a unified framework for sign language understanding.

More information about the model and our implementation is available in the [Uni-Sign repository](https://github.com/zeleznyt/Uni-Sign).

## Resources

- [Demo source code](https://github.com/JSALT2024/slt-demo/tree/uni-sign-EP)
- [Pose estimation and preprocessing](https://github.com/JSALT2024/PoseEstimation)
- [Uni-Sign implementation](https://github.com/zeleznyt/Uni-Sign)
- **Uni-Sign:** *Uni-Sign: Toward Unified Sign Language Understanding at Scale*, Li et al., ICLR 2025

## Acknowledgements

This demonstrator was developed at the **University of West Bohemia (ZČU), Department of Cybernetics, Computer Vision group**.

Development was supported by the **2026 ZČU internal mini-project programme for the development and wider use of artificial intelligence**.

The system builds upon the **Uni-Sign** framework. We thank its authors and the open-source sign language research community for making their work publicly available.
"""



with gr.Blocks(title="Sign Language Translation", css=custom_css, theme=gr.themes.Default(primary_hue="amber", neutral_hue="neutral")) as app:
    
    current_video = gr.State("")

    with gr.Column(elem_id="main-layout"):
        
        # App Header Row: Left Badge (ZČU Logo), Title & Subtitle, Right Badge (FAV ZČU Logo)
        gr.HTML(f"""
        <div class="app-header-container">
            <div class="header-badge header-badge-left">
                <a href="https://www.zcu.cz" target="_blank" rel="noopener noreferrer" title="Západočeská univerzita v Plzni">
                    <img src="{zcu_logo_src}" alt="Západočeská univerzita v Plzni" class="header-badge-img" />
                </a>
            </div>
            <div class="header-titles">
                <h1 class="app-title">Sign Language to Text Translation</h1>
                <h2 class="app-subtitle">Upload an ASL video and get a text translation.</h2>
            </div>
            <div class="header-badge header-badge-right">
                <a href="https://fav.zcu.cz" target="_blank" rel="noopener noreferrer" title="Fakulta aplikovaných věd ZČU">
                    <img src="{fav_logo_src}" alt="Fakulta aplikovaných věd ZČU" class="header-badge-img" />
                </a>
            </div>
        </div>
        """, elem_classes=["app-header-html"])

        # Card 1: Upload Video Box (Compact Dropzone & Info Row)
        with gr.Column(elem_classes=["ui-card"]):
            with gr.Row(elem_classes=["card-header-row"]):
                with gr.Column(scale=1, min_width=0):
                    gr.Markdown("<h3 class='card-title'>Upload video</h3>")
                with gr.Column(scale=0, min_width=160, elem_classes=["record-btn-col"]):
                    record_yourself_btn = gr.Button(
                        "📹 RECORD YOURSELF",
                        variant="secondary",
                        elem_classes=["record-yourself-btn"],
                        visible=True,
                    )
            
            # Compact file dropzone
            upload_file = gr.File(
                label="Upload Video",
                file_types=["video"],
                file_count="single",
                show_label=False,
                elem_classes=["compact-dropzone"]
            )
            
            # Active video status row: Video name on left, action buttons aligned to the right
            with gr.Row(visible=False, elem_classes=["video-info-row"]) as video_info_row:
                video_name_md = gr.Markdown("<div class='video-name-badge'><span>video.mp4</span></div>", elem_classes=["video-name-col"])
                preview_btn = gr.Button("🎬 Preview & Trim Video", variant="secondary", elem_classes=["preview-modal-btn"])
                change_video_btn = gr.Button("✕ Remove", variant="secondary", elem_classes=["change-vid-btn"])

        # Standalone Translate Action Button (hidden until a video is uploaded or selected)
        submit_btn = gr.Button("Translate", variant="primary", elem_classes=["translate-btn"], visible=False)
        
        # Card 2: Translation Result Box (hidden until Translate is clicked)
        with gr.Column(elem_classes=["ui-card", "translation-card"], elem_id="translation-card", visible=False) as translation_card:
            gr.Markdown("<h3 class='card-title'>Translation</h3>")
            translation_display = gr.HTML(value="", elem_id="translation-display", elem_classes=["translation-display-html"])
        
        # Card 3: Examples Box (6 videos in 3 columns x 2 rows)
        with gr.Column(elem_classes=["ui-card"]):
            gr.Markdown("<h3 class='card-title'>Examples</h3>")
            with gr.Row(elem_classes=["examples-row"]):
                with gr.Column(scale=1):
                    gr.HTML("<div class='example-title'>Book</div>")
                    gr.Video(value=example_videos[0], interactive=False, show_label=False, autoplay=False, height=160, mirror_webcam=False)
                    btn_ex1 = gr.Button("Use Example 1", variant="secondary", elem_classes=["example-btn"])
                with gr.Column(scale=1):
                    gr.HTML("<div class='example-title'>Deaf</div>")
                    gr.Video(value=example_videos[1], interactive=False, show_label=False, autoplay=False, height=160, mirror_webcam=False)
                    btn_ex2 = gr.Button("Use Example 2", variant="secondary", elem_classes=["example-btn"])
                with gr.Column(scale=1):
                    gr.HTML("<div class='example-title'>Help</div>")
                    gr.Video(value=example_videos[2], interactive=False, show_label=False, autoplay=False, height=160, mirror_webcam=False)
                    btn_ex3 = gr.Button("Use Example 3", variant="secondary", elem_classes=["example-btn"])
            
            with gr.Row(elem_classes=["examples-row", "examples-row-second"]):
                with gr.Column(scale=1):
                    gr.HTML("<div class='example-title'>Fine</div>")
                    gr.Video(value=example_videos[3], interactive=False, show_label=False, autoplay=False, height=160, mirror_webcam=False)
                    btn_ex4 = gr.Button("Use Example 4", variant="secondary", elem_classes=["example-btn"])
                with gr.Column(scale=1):
                    gr.HTML("<div class='example-title'>Woman</div>")
                    gr.Video(value=example_videos[4], interactive=False, show_label=False, autoplay=False, height=160, mirror_webcam=False)
                    btn_ex5 = gr.Button("Use Example 5", variant="secondary", elem_classes=["example-btn"])
                with gr.Column(scale=1):
                    gr.HTML("<div class='example-title'>No</div>")
                    gr.Video(value=example_videos[5], interactive=False, show_label=False, autoplay=False, height=160, mirror_webcam=False)
                    btn_ex6 = gr.Button("Use Example 6", variant="secondary", elem_classes=["example-btn"])

        # Card 4: Information Box
        with gr.Column(elem_classes=["ui-card"]):
            gr.Markdown("<h3 class='card-title'>Information</h3>")
            gr.Markdown(INFO_MARKDOWN, elem_classes=["info-markdown"])
            
    # Fullscreen Floating Modal Window for Video Preview & Trimming
    with gr.Column(elem_classes=["modal-overlay"], visible=False) as preview_modal:
        with gr.Column(elem_classes=["modal-dialog-card"]):
            with gr.Row(elem_classes=["modal-header-row"]):
                with gr.Column(scale=1, min_width=0):
                    gr.Markdown("<h3 class='modal-title'>Video Recording, Preview & Trimming</h3>")
                with gr.Column(scale=0, min_width=36, elem_classes=["modal-close-col"]):
                    modal_close_top = gr.Button("✕", size="sm", min_width=36, elem_classes=["modal-close-icon"])
            
            modal_video = gr.Video(
                interactive=True,
                show_label=False,
                sources=["upload", "webcam"],
                mirror_webcam=True,
                elem_classes=["modal-video-player"],
            )
            
            with gr.Column(elem_classes=["modal-footer-col"]):
                modal_flip_btn = gr.Button("⇄ Flip Horizontally (Mirror)", variant="secondary", elem_classes=["modal-flip-btn"])
                modal_save_btn = gr.Button("✓ Save & Use Video", variant="primary", elem_classes=["modal-done-btn"])

    def start_translating_ui():
        spinner_html = """<div class="translation-content-box"><div class="loading-container"><div class="pulse-spinner"></div></div></div>"""
        return gr.update(visible=True), spinner_html

    def finish_translating(video_path):
        if not video_path:
            return """<div class="translation-content-box"><div style="color: #dba70e; font-weight: 600; font-size: 15px; text-align: center;">Please select or upload a video first.</div></div>"""
        raw_result = process_video(video_path)
        clean_result = str(raw_result).strip()
        if clean_result.startswith("Error") or "error" in clean_result.lower():
            return f"""<div class="translation-content-box"><div style="color: #ff6b6b; font-weight: 600; font-size: 14px; text-align: center;">{clean_result}</div></div>"""
        return f"""<div class="translation-content-box"><div class="translation-text">{clean_result}</div></div>"""

    # Upload file event
    upload_file.upload(
        fn=handle_file_upload,
        inputs=upload_file,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
    )

    # Remove / Change video event
    change_video_btn.click(
        fn=handle_remove_video,
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
    )

    # Smooth scroll to top JavaScript helper for example selections
    scroll_top_js = "() => { window.scrollTo({ top: 0, behavior: 'smooth' }); }"

    # Examples click events with smooth auto-scroll to top
    btn_ex1.click(
        fn=lambda: handle_select_example(example_videos[0]),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex2.click(
        fn=lambda: handle_select_example(example_videos[1]),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex3.click(
        fn=lambda: handle_select_example(example_videos[2]),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex4.click(
        fn=lambda: handle_select_example(example_videos[3]),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex5.click(
        fn=lambda: handle_select_example(example_videos[4]),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex6.click(
        fn=lambda: handle_select_example(example_videos[5]),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )

    # Open Modal event from Preview & Trim button
    preview_btn.click(
        fn=open_modal,
        inputs=current_video,
        outputs=[preview_modal, modal_video],
    )

    # Open Modal event from Record Yourself button
    record_yourself_btn.click(
        fn=open_modal_for_recording,
        inputs=None,
        outputs=[preview_modal, modal_video],
    )

    # Flip / Mirror video horizontally
    modal_flip_btn.click(
        fn=flip_video_horizontal,
        inputs=modal_video,
        outputs=modal_video,
    )

    # Save & Use Video
    modal_save_btn.click(
        fn=close_modal_and_save,
        inputs=[modal_video, current_video],
        outputs=[preview_modal, current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
    )

    # Close / Cancel Modal event
    modal_close_top.click(
        fn=cancel_modal,
        inputs=current_video,
        outputs=[preview_modal, current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
    )

    # Translate event
    submit_btn.click(
        fn=start_translating_ui,
        inputs=None,
        outputs=[translation_card, translation_display],
    ).then(
        fn=finish_translating,
        inputs=current_video,
        outputs=translation_display,
    )



if __name__ == "__main__":
    app.launch()
