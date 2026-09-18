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

# Prepare base64-encoded logo for the fixed institution badge
logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fav_zcu_logo.png")
logo_src = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        logo_src = f"data:image/png;base64,{logo_b64}"

# Paths to the 3 example videos (first 3 from wlasl, re-encoded to H.264)
example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "wlasl")
if not os.path.exists(example_dir):
    example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DELETE", "examples", "wlasl")

example_videos = [
    os.path.join(example_dir, "07093.mp4"),
    os.path.join(example_dir, "32167.mp4"),
    os.path.join(example_dir, "63415.mp4"),
]

# Load CSS from external style.css file
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    custom_css = f.read()


with gr.Blocks(title="Sign Language Translation", css=custom_css, theme=gr.themes.Default(primary_hue="amber", neutral_hue="neutral")) as app:
    
    gr.Markdown("<h1>Sign Language to Text Translation</h1>")
    gr.Markdown("<h3 class='page-subtitle'>Upload an ASL video and get a text translation.</h3>")
    
    current_video = gr.State("")

    with gr.Column(elem_id="main-layout"):
        
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
        
        # Card 3: Examples Box (3 videos side-by-side)
        with gr.Column(elem_classes=["ui-card"]):
            gr.Markdown("<h3 class='card-title'>Examples</h3>")
            with gr.Row(elem_classes=["examples-row"]):
                with gr.Column(scale=1):
                    gr.Video(value=example_videos[0], interactive=False, show_label=False, autoplay=False, height=160)
                    btn_ex1 = gr.Button("Use Example 1", variant="secondary", elem_classes=["example-btn"])
                with gr.Column(scale=1):
                    gr.Video(value=example_videos[1], interactive=False, show_label=False, autoplay=False, height=160)
                    btn_ex2 = gr.Button("Use Example 2", variant="secondary", elem_classes=["example-btn"])
                with gr.Column(scale=1):
                    gr.Video(value=example_videos[2], interactive=False, show_label=False, autoplay=False, height=160)
                    btn_ex3 = gr.Button("Use Example 3", variant="secondary", elem_classes=["example-btn"])

        # Card 4: Information Box
        with gr.Column(elem_classes=["ui-card"]):
            gr.Markdown("<h3 class='card-title'>Information</h3>")
            gr.HTML("""
            <div class="info-content">
                <div class="info-text">
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                </div>
                <a href="#" class="advanced-demo-link">
                    <span class="adv-line adv-line-1">EXPLORE</span>
                    <span class="adv-line adv-line-2">ADVANCED</span>
                    <span class="adv-line adv-line-3">⭐DEMO⭐</span>
                </a>
            </div>
            """)
            
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
                sources=["webcam", "upload"],
                elem_classes=["modal-video-player"],
            )
            
            with gr.Row(elem_classes=["modal-footer-row"]):
                modal_save_btn = gr.Button("✓ Save & Use Video", variant="primary", elem_classes=["modal-done-btn"])

    def start_translating_ui():
        return gr.update(visible=True), ""

    def finish_translating(video_path):
        #if not video_path:
        #    return """<div class="translation-content-box"><div style="color: #ff6b6b; font-weight: 600; font-size: 15px; text-align: center;">Please upload or select a video first.</div></div>"""
        raw_result = process_video(video_path)
        clean_result = str(raw_result).strip()
        #if clean_result.startswith("Error") or "error" in clean_result.lower():
        #    return f"""<div class="translation-content-box"><div style="color: #ff6b6b; font-weight: 600; font-size: 15px; text-align: center;">{clean_result}</div></div>"""
        # věřím že není potřeba protože vždy musí být path, a errory budu řešit později
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
        fn=lambda: handle_select_example(example_videos[0], "Example 1"),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex2.click(
        fn=lambda: handle_select_example(example_videos[1], "Example 2"),
        inputs=None,
        outputs=[current_video, upload_file, record_yourself_btn, video_info_row, video_name_md, submit_btn, translation_card, modal_video],
        js=scroll_top_js,
    )
    btn_ex3.click(
        fn=lambda: handle_select_example(example_videos[2], "Example 3"),
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

    # Fixed floating badge in bottom-right corner with full-width logo
    gr.HTML(f"""
    <div class="support-badge">
        <span class="support-badge-text">Research supported by:</span>
        <a href="https://fav.zcu.cz" target="_blank" rel="noopener noreferrer">
            <img src="{logo_src}" alt="Faculty of Applied Sciences, University of West Bohemia in Pilsen" class="support-badge-img" />
        </a>
    </div>
    """)

if __name__ == "__main__":
    app.launch()
