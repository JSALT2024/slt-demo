"""
handle_gradio.py
Manages Gradio event handlers, UI state transitions, and user interactions.
"""

import os
import tempfile
import subprocess
import shutil
import gradio as gr
from backend import process_input

# State to track whether webcam recording mode is actively open in the modal
_is_recording_mode = False

# ==============================================================================
# 1. FFMPEG & GRADIO PLAYABILITY SETUP
# ==============================================================================

def setup_ffmpeg():
    """Ensure ffmpeg from imageio_ffmpeg is copied as ffmpeg.exe and added to PATH."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        ffmpeg_target = os.path.join(ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not os.path.exists(ffmpeg_target):
            try:
                shutil.copyfile(ffmpeg_exe, ffmpeg_target)
            except Exception:
                pass
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        print(f"Warning setting up ffmpeg: {e}")


def get_ffmpeg_bin():
    """Resolves a working ffmpeg binary executable path."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    return "ffmpeg"


def patch_gradio_playable():
    """Patch Gradio's video_is_playable to prevent FFExecutableNotFoundError when ffprobe is not installed."""
    try:
        import gradio.processing_utils as gr_proc
        _orig_playable = gr_proc.video_is_playable

        def _safe_playable(video_filepath):
            try:
                return _orig_playable(video_filepath)
            except Exception:
                return True

        gr_proc.video_is_playable = _safe_playable
    except Exception:
        pass


# Run setup immediately on module import
setup_ffmpeg()
patch_gradio_playable()


# ==============================================================================
# 2. VIDEO PATH & CODEC CONVERSION UTILITIES
# ==============================================================================

def extract_video_path(val):
    """Extracts a valid filesystem path string from any Gradio video object, dict, or file wrapper."""
    if not val:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        if "video" in val and val["video"]:
            return extract_video_path(val["video"])
        if "path" in val and val["path"]:
            return extract_video_path(val["path"])
        if "url" in val and val["url"]:
            return str(val["url"])
    if hasattr(val, "path") and val.path:
        return str(val.path)
    if hasattr(val, "name") and val.name:
        return str(val.name)
    if hasattr(val, "video") and val.video:
        return extract_video_path(val.video)
    return str(val)


def ensure_web_compatible_video(video_path):
    """Ensures input video is encoded in browser/OpenCV compatible H.264 format."""
    path_str = extract_video_path(video_path)
    if not path_str or not os.path.exists(path_str):
        return path_str

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
        ffmpeg_bin = get_ffmpeg_bin()
        mtime = int(os.path.getmtime(path_str))
        clean_name = os.path.basename(path_str)
        import re
        clean_name = re.sub(r'(_flipped(_\d+)?)', '', clean_name)
        clean_name = re.sub(r'(_\d+_h264)', '', clean_name)
        base, _ = os.path.splitext(clean_name)

        out_dir = os.path.join(tempfile.gettempdir(), "slt_compat_videos", str(mtime))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}.mp4")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            cmd = [
                ffmpeg_bin, "-y",
                "-i", path_str,
                "-r", "30",
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


def is_webcam_video(path_str: str) -> bool:
    """Determines if the video is from a webcam recording."""
    if not path_str:
        return False
    base = os.path.basename(path_str).lower()
    return base in ["webkamera.mp4", "input_video.mp4", "input_video.webm"] or base.startswith("webkamera")


def format_video_display_name(video_input, is_webcam: bool = False) -> str:
    """Formats a video filename for UI display, removing temp/flip suffixes and capping at 20 chars max + '...'."""
    path_str = extract_video_path(video_input)
    if not path_str:
        return ""
    if is_webcam or is_webcam_video(path_str):
        return "webkamera.mp4"
    fname = os.path.basename(path_str)
    import re
    clean_name = re.sub(r'(_flipped(_\d+)?)', '', fname)
    clean_name = re.sub(r'(_\d+_h264)', '', clean_name)
    if len(clean_name) > 20:
        return clean_name[:20] + "..."
    return clean_name


def process_video(input_video_path, progress=None):
    """Prepares and translates an input video using the model backend."""
    if not input_video_path:
        return "Please upload or select a video first.", ""
    compatible_path = ensure_web_compatible_video(input_video_path)
    translation, keypoints_video_path = process_input(compatible_path, progress=progress)
    return translation, keypoints_video_path


# ==============================================================================
# 3. GRADIO EVENT HANDLERS FOR VIDEO INPUT
# ==============================================================================

def handle_file_upload(file):
    """Handles video file upload, transcoding to H.264 and updating UI state."""
    global _is_recording_mode
    _is_recording_mode = False
    if not file:
        return (
            "",                                                          # current_video
            gr.update(visible=True, value=None),                         # upload_file
            gr.update(visible=True),                                     # record_yourself_btn
            gr.update(visible=False),                                    # video_info_row
            gr.update(value=""),                                         # video_name_md
            gr.update(visible=False),                                    # submit_btn
            gr.update(visible=False),                                    # translation_card
            gr.update(value=None),                                       # modal_video
        )
    compatible = ensure_web_compatible_video(file)
    display_name = format_video_display_name(file if isinstance(file, str) or hasattr(file, 'name') else compatible)
    return (
        compatible,                                                  # current_video
        gr.update(visible=False),                                    # upload_file
        gr.update(visible=False),                                    # record_yourself_btn
        gr.update(visible=True),                                     # video_info_row
        gr.update(value=f"<div class='video-name-badge'><b>{display_name}</b></div>"), # video_name_md
        gr.update(visible=True),                                     # submit_btn
        gr.update(visible=False),                                    # translation_card
        gr.update(value=compatible),                                 # modal_video
    )


def handle_remove_video():
    """Clears the currently active video and resets the upload box."""
    global _is_recording_mode
    _is_recording_mode = False
    return (
        "",                                                          # current_video
        gr.update(visible=True, value=None),                         # upload_file
        gr.update(visible=True),                                     # record_yourself_btn
        gr.update(visible=False),                                    # video_info_row
        gr.update(value=""),                                         # video_name_md
        gr.update(visible=False),                                    # submit_btn
        gr.update(visible=False),                                    # translation_card
        gr.update(value=None),                                       # modal_video
    )


def handle_select_example(ex_path, ex_label=None):
    """Selects an example video, updates the active video badge, and enables translation."""
    global _is_recording_mode
    _is_recording_mode = False
    display_name = format_video_display_name(ex_path)
    return (
        ex_path,                                                     # current_video
        gr.update(visible=False),                                    # upload_file
        gr.update(visible=False),                                    # record_yourself_btn
        gr.update(visible=True),                                     # video_info_row
        gr.update(value=f"<div class='video-name-badge'><b>{display_name}</b></div>"), # video_name_md
        gr.update(visible=True),                                     # submit_btn
        gr.update(visible=False),                                    # translation_card
        gr.update(value=ex_path),                                    # modal_video
    )


def open_modal(curr_vid):
    """Opens the preview & trimming modal with the current video loaded."""
    global _is_recording_mode
    _is_recording_mode = False
    return gr.update(visible=True), gr.update(value=curr_vid)


def open_modal_for_recording():
    """Opens the preview modal in webcam recording mode."""
    global _is_recording_mode
    _is_recording_mode = True
    return gr.update(visible=True), gr.update(value=None)


def close_modal_and_save(mod_vid, curr_vid):
    """Saves recorded or trimmed video from modal, transcoding to H.264 if needed."""
    global _is_recording_mode
    target_path = extract_video_path(mod_vid)
    if not target_path or not os.path.exists(target_path):
        target_path = extract_video_path(curr_vid)

    if target_path and os.path.exists(target_path):
        is_webcam = _is_recording_mode or (not curr_vid) or is_webcam_video(target_path)
        _is_recording_mode = False

        if is_webcam:
            import time
            out_dir = os.path.join(tempfile.gettempdir(), "slt_webcam_videos", str(int(time.time() * 1000)))
            os.makedirs(out_dir, exist_ok=True)
            webcam_path = os.path.join(out_dir, "webkamera.mp4")
            ffmpeg_bin = get_ffmpeg_bin()
            cmd = [
                ffmpeg_bin, "-y",
                "-i", target_path,
                "-r", "30",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                webcam_path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(webcam_path) and os.path.getsize(webcam_path) > 0:
                    target_path = webcam_path
            except Exception as e:
                print(f"Error transcoding webcam video: {e}")

        compat_path = ensure_web_compatible_video(target_path)
        display_name = format_video_display_name(compat_path, is_webcam=is_webcam)
        return (
            gr.update(visible=False),                                                 # preview_modal
            compat_path,                                                              # current_video
            gr.update(visible=False),                                                 # upload_file
            gr.update(visible=False),                                                 # record_yourself_btn
            gr.update(visible=True),                                                  # video_info_row
            gr.update(value=f"<div class='video-name-badge'><b>{display_name}</b></div>"),   # video_name_md
            gr.update(visible=True),                                                  # submit_btn
            gr.update(visible=False),                                                 # translation_card
            compat_path,                                                              # modal_video
        )
    else:
        _is_recording_mode = False
        return (
            gr.update(visible=False),                                                 # preview_modal
            "",                                                                       # current_video
            gr.update(visible=True, value=None),                                      # upload_file
            gr.update(visible=True),                                                  # record_yourself_btn
            gr.update(visible=False),                                                 # video_info_row
            gr.update(value=""),                                                      # video_name_md
            gr.update(visible=False),                                                 # submit_btn
            gr.update(visible=False),                                                 # translation_card
            None,                                                                     # modal_video
        )


def cancel_modal(curr_vid):
    """Closes modal without modifying current video."""
    global _is_recording_mode
    _is_recording_mode = False
    target_path = extract_video_path(curr_vid)
    if target_path and os.path.exists(target_path):
        display_name = format_video_display_name(target_path)
        return (
            gr.update(visible=False),                                                 # preview_modal
            target_path,                                                              # current_video
            gr.update(visible=False),                                                 # upload_file
            gr.update(visible=False),                                                 # record_yourself_btn
            gr.update(visible=True),                                                  # video_info_row
            gr.update(value=f"<div class='video-name-badge'><b>{display_name}</b></div>"),   # video_name_md
            gr.update(visible=True),                                                  # submit_btn
            gr.update(visible=False),                                                 # translation_card
            target_path,                                                              # modal_video
        )
    else:
        return (
            gr.update(visible=False),                                                 # preview_modal
            "",                                                                       # current_video
            gr.update(visible=True, value=None),                                      # upload_file
            gr.update(visible=True),                                                  # record_yourself_btn
            gr.update(visible=False),                                                 # video_info_row
            gr.update(value=""),                                                      # video_name_md
            gr.update(visible=False),                                                 # submit_btn
            gr.update(visible=False),                                                 # translation_card
            None,                                                                     # modal_video
        )


def open_keypoints_modal(keypoints_video):
    """Opens the keypoints visualization modal window."""
    if not keypoints_video or not os.path.exists(keypoints_video):
        return gr.update(visible=False), None
    return gr.update(visible=True), keypoints_video


def close_keypoints_modal():
    """Closes the keypoints visualization modal window."""
    return gr.update(visible=False), None
