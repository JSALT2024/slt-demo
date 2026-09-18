import gradio as gr
import os
import base64
import tempfile
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Ensure ffmpeg from imageio_ffmpeg is in PATH for Gradio video player and processing
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

from backend import process_input

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


def ensure_web_compatible_video(video_path):
    """Ensures input video is encoded in browser/OpenCV compatible H.264 format."""
    if not video_path:
        return video_path
    
    if hasattr(video_path, "path"):
        path_str = video_path.path
    elif isinstance(video_path, dict) and "path" in video_path:
        path_str = video_path["path"]
    elif isinstance(video_path, str):
        path_str = video_path
    else:
        path_str = str(video_path)

    if not os.path.exists(path_str):
        return video_path

    # Check if already H.264
    try:
        import cv2
        cap = cv2.VideoCapture(path_str)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).lower()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fourcc_str in ['h264', 'avc1'] and frame_count > 0:
            return path_str
    except Exception:
        pass

    # Transcode to standard H.264 using ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        out_dir = os.path.join(tempfile.gettempdir(), "slt_compat_videos")
        os.makedirs(out_dir, exist_ok=True)
        mtime = int(os.path.getmtime(path_str))
        base = os.path.splitext(os.path.basename(path_str))[0]
        out_path = os.path.join(out_dir, f"{base}_{mtime}_h264.mp4")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            cmd = [
                ffmpeg_bin, "-y",
                "-i", path_str,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"FFmpeg conversion warning: {res.stderr}")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        print(f"Video conversion error: {e}")

    return path_str


def process_video(input_video_path):
    if not input_video_path:
        return "Please upload or select a video first."
    # Ensure video is in standard web/OpenCV compatible H.264 format
    compatible_path = ensure_web_compatible_video(input_video_path)
    translation = process_input(compatible_path)
    return translation

# Custom CSS for dark palette (#221f1f), golden accents (#dba70e), clean text (#f5f5f5), and modular cards
custom_css = """
body, html, gradio-app {
    background-color: #221f1f !important;
    color: #f5f5f5 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}
.gradio-container {
    background-color: #221f1f !important;
    border: none !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

#main-layout {
    max-width: 800px !important;
    width: 100% !important;
    margin: 10px auto 40px auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

/* Individual Modular Cards */
.ui-card {
    background-color: #2d2929 !important;
    border: 1px solid rgba(219, 167, 14, 0.4) !important;
    border-radius: 14px !important;
    padding: 20px 24px !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.card-title {
    color: #dba70e !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    margin: 0 0 12px 0 !important;
    text-transform: uppercase !important;
}

h1 {
    text-align: center;
    color: #dba70e !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
h3.page-subtitle {
    text-align: center;
    color: #f5f5f5 !important;
    font-weight: 400 !important;
    opacity: 0.9;
    margin-top: 0;
    margin-bottom: 20px;
}

/* Primary Translate Button */
.translate-btn, button.primary {
    background: linear-gradient(135deg, #dba70e 0%, #be9007 100%) !important;
    color: #221f1f !important;
    font-weight: 800 !important;
    font-size: 18px !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 10px !important;
    height: 48px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(219, 167, 14, 0.35) !important;
    transition: all 0.2s ease !important;
    margin: 4px 0 !important;
}
.translate-btn:hover, button.primary:hover {
    background: linear-gradient(135deg, #e8b625 0%, #dba70e 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(219, 167, 14, 0.5) !important;
}

/* Form component styling */
textarea, input[type="text"] {
    background-color: #1a1818 !important;
    color: #f5f5f5 !important;
    border-color: rgba(219, 167, 14, 0.3) !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #dba70e !important;
    box-shadow: 0 0 0 2px rgba(219, 167, 14, 0.25) !important;
}

/* Ensure Examples are strictly 3 side-by-side columns */
.examples-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 14px !important;
    width: 100% !important;
}
.examples-row > div {
    flex: 1 1 0px !important;
    min-width: 0 !important;
    width: 32% !important;
}

.example-btn {
    background: #383333 !important;
    color: #f5f5f5 !important;
    border: 1px solid rgba(219, 167, 14, 0.35) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    margin-top: 8px !important;
    padding: 8px 10px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.example-btn:hover {
    background: #dba70e !important;
    color: #221f1f !important;
    border-color: #dba70e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(219, 167, 14, 0.35) !important;
}

/* Floating support badge in bottom-right corner */
.support-badge {
    position: fixed;
    bottom: 22px;
    right: 22px;
    z-index: 9999;
    width: 260px;
    box-sizing: border-box;
    background-color: rgba(34, 31, 31, 0.92);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(219, 167, 14, 0.45);
    border-radius: 12px;
    padding: 10px 14px 12px 14px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.support-badge:hover {
    transform: translateY(-3px);
    border-color: #dba70e;
    box-shadow: 0 8px 24px rgba(219, 167, 14, 0.3);
}
.support-badge-text {
    color: #dba70e;
    font-size: 13.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    width: 100%;
    text-align: center;
    display: block;
    margin: 0 0 2px 0;
}
.support-badge a {
    display: block;
    width: 100%;
}
.support-badge-img {
    width: 100% !important;
    height: auto !important;
    display: block;
    object-fit: contain;
}

@media (max-width: 600px) {
    .examples-row {
        flex-direction: column !important;
    }
    .examples-row > div {
        width: 100% !important;
    }
    .support-badge {
        bottom: 12px;
        right: 12px;
        width: 180px;
        padding: 8px 10px;
    }
    .support-badge-text {
        font-size: 9px;
        letter-spacing: 1px;
    }
}
"""
# Custom script injected in <head> to silently suppress transient "Video not playable" popups during upload conversion
custom_head = """
<script>
(function() {
    // Intercept video error events before Gradio Svelte handler triggers toast
    window.addEventListener('error', function(e) {
        if (e.target && (e.target.tagName === 'VIDEO' || e.target.tagName === 'SOURCE')) {
            e.stopImmediatePropagation();
            e.preventDefault();
        }
    }, true);

    // Auto-dismiss any 'not playable' toast popups that might appear
    function hidePlayableErrors() {
        var toasts = document.querySelectorAll('.toast, .toast-wrap, [class*="toast"], .error, [data-testid="error-message"]');
        toasts.forEach(function(t) {
            var txt = (t.innerText || t.textContent || '').toLowerCase();
            if (txt.indexOf('playable') !== -1 || txt.indexOf('not playable') !== -1 || txt.indexOf('video error') !== -1) {
                t.style.display = 'none';
            }
        });
    }

    var observer = new MutationObserver(function(mutations) {
        hidePlayableErrors();
    });

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            observer.observe(document.body || document.documentElement, { childList: true, subtree: true });
        });
    } else {
        observer.observe(document.body || document.documentElement, { childList: true, subtree: true });
    }
})();
</script>
"""

with gr.Blocks(title="Sign Language Translation", css=custom_css, head=custom_head, theme=gr.themes.Default(primary_hue="amber", neutral_hue="neutral")) as app:
    
    gr.Markdown("<h1>Sign Language to Text Translation</h1>")
    gr.Markdown("<h3 class='page-subtitle'>Upload an ASL video and get a text translation.</h3>")
    
    with gr.Column(elem_id="main-layout"):
        
        # Card 1: Upload Video Box
        with gr.Column(elem_classes=["ui-card"]):
            gr.Markdown("<h3 class='card-title'>Upload video</h3>")
            video_input = gr.Video(show_label=False, format="mp4")
            
        # Standalone Translate Action Button
        submit_btn = gr.Button("Translate", variant="primary", elem_classes=["translate-btn"])
        
        # Card 2: Translation Result Box
        with gr.Column(elem_classes=["ui-card"]):
            gr.Markdown("<h3 class='card-title'>Translation</h3>")
            text_output = gr.Textbox(show_label=False, placeholder="Translation will appear here...", lines=3)
        
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
            
    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=text_output,
    )

    # Automatically transcode any uploaded video (e.g. FMP4/AVI) to web-friendly H.264 on upload
    video_input.upload(
        fn=ensure_web_compatible_video,
        inputs=video_input,
        outputs=video_input,
    )

    # Click handlers to load the example video into the main upload box
    btn_ex1.click(fn=lambda: example_videos[0], inputs=None, outputs=video_input)
    btn_ex2.click(fn=lambda: example_videos[1], inputs=None, outputs=video_input)
    btn_ex3.click(fn=lambda: example_videos[2], inputs=None, outputs=video_input)

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

