"""
visualize_pose.py
Renders MediaPipe pose, hand, and facial mesh keypoints with colored skeletons,
facial detail differentiation (green details, blue mesh), and bounding boxes
(blue for face, red for right hand, purple for left hand) onto video frames.
Automatically upscales low-resolution videos to crisp 1080p equivalent resolution
so all points, lines, and boxes are rendered with anti-aliasing without pixelation.
Encodes the output as a high-definition H.264 MP4 video for browser playback.
"""

import os
import tempfile
import time
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg

# ==============================================================================
# 1. LANDMARK SKELETON CONNECTIONS & FACIAL DETAIL INDICES
# ==============================================================================

# MediaPipe Pose landmark connections (torso & arms)
POSE_CONNECTIONS = [
    # Torso & Shoulders
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15),
    # Right Arm
    (12, 14), (14, 16),
]

# MediaPipe Hand landmark connections (21 points)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky finger
    (0, 17)                                # Palm base
]

# MediaPipe Face Mesh landmark indices for facial details (eyes, eyebrows, lips, nose)
FACEMESH_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]
FACEMESH_LEFT_EYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    468, 469, 470, 471, 472
]
FACEMESH_LEFT_EYEBROW = [276, 283, 282, 295, 300, 293, 334, 296, 336, 285]
FACEMESH_RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    473, 474, 475, 476, 477
]
FACEMESH_RIGHT_EYEBROW = [46, 53, 52, 65, 70, 63, 105, 66, 107, 55]
FACEMESH_NOSE = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 275, 45, 220, 115]

FACE_DETAIL_INDICES = set(
    FACEMESH_LIPS + FACEMESH_LEFT_EYE + FACEMESH_LEFT_EYEBROW +
    FACEMESH_RIGHT_EYE + FACEMESH_RIGHT_EYEBROW + FACEMESH_NOSE
)

# Canonical Colors (RGB) matching compare_jsons.py style:
COLOR_POSE = (0, 230, 118)            # Green for body skeleton connections (spojení těla) & joints
COLOR_POSE_GREEN = (0, 230, 118)      # Green for key body joints (shoulders, elbows, hips)
COLOR_FACE_BASE = (0, 136, 255)       # Blue for face mesh
COLOR_FACE_DETAILS = (0, 230, 118)    # Green for facial details (eyes, nose, mouth)
COLOR_FACE_BOX = (0, 136, 255)        # Blue bounding box for face
COLOR_RIGHT_HAND = (255, 51, 51)      # Red for right hand
COLOR_RIGHT_BOX = (255, 51, 51)       # Red bounding box for right hand
COLOR_LEFT_HAND = (181, 72, 255)      # Purple / Violet for left hand
COLOR_LEFT_BOX = (181, 72, 255)       # Purple / Violet bounding box for left hand
COLOR_JOINT = (255, 255, 255)         # Joint center highlight (White)

POSE_GREEN_JOINTS = {11, 12, 13, 14, 23, 24}  # Shoulders, elbows, hips


def hex_to_rgb(hex_code) -> tuple:
    """Converts hex string like '#00DCFF' or RGB tuple/list to RGB tuple (R, G, B)."""
    if isinstance(hex_code, (tuple, list)):
        return tuple(int(c) for c in hex_code[:3])
    h = str(hex_code).lstrip('#')
    if len(h) == 6:
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def get_bbox(kps: dict, bbox_key: str, landmarks_key: str):
    """
    Extracts or computes bounding box [x_min, y_min, x_max, y_max].
    Prefers pre-computed bounding box from YOLO/MediaPipe crop if valid;
    otherwise computes bounding box from landmarks with 15% margin.
    """
    bbox = kps.get(bbox_key)
    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(float(c))) for c in bbox[:4]]
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]

    pts = kps.get(landmarks_key, [])
    if len(pts) > 0:
        arr = np.array(pts, dtype=float)
        x_min = int(round(np.min(arr[:, 0])))
        y_min = int(round(np.min(arr[:, 1])))
        x_max = int(round(np.max(arr[:, 0])))
        y_max = int(round(np.max(arr[:, 1])))
        if x_max > x_min and y_max > y_min:
            pad_x = max(4, int((x_max - x_min) * 0.15))
            pad_y = max(4, int((y_max - y_min) * 0.15))
            return [max(0, x_min - pad_x), max(0, y_min - pad_y), x_max + pad_x, y_max + pad_y]

    return None


# ==============================================================================
# 2. FRAME DRAWING LOGIC (1080p UPSCALE & CRISP VECTOR RENDERING)
# ==============================================================================

def draw_keypoints_frame(
    frame: np.ndarray,
    kps: dict,
    target_width: int = None,
    target_height: int = None,
    pose_color=COLOR_POSE,
    pose_opacity: float = 0.88,
    face_color=COLOR_FACE_BASE,
    face_details_color=COLOR_FACE_DETAILS,
    face_box_color=COLOR_FACE_BOX,
    face_opacity: float = 0.88,
    left_hand_color=COLOR_LEFT_HAND,
    left_hand_box_color=COLOR_LEFT_BOX,
    left_hand_opacity: float = 0.88,
    right_hand_color=COLOR_RIGHT_HAND,
    right_hand_box_color=COLOR_RIGHT_BOX,
    right_hand_opacity: float = 0.88,
    pose_thickness: int = None,
    pose_radius: int = None,
    face_radius: int = None,
    hand_thickness: int = None,
    hand_radius: int = None,
    box_thickness: int = None,
    draw_boxes: bool = True,
) -> np.ndarray:
    """
    Overlays colored keypoint circles, skeleton connection lines, and bounding boxes
    on an RGB image frame. If target_width and target_height are specified, the frame
    is smoothly upscaled using Lanczos interpolation, and all vector elements are drawn
    at full resolution with anti-aliasing (LINE_AA) for sharp, unpixelated results.
    """
    orig_h, orig_w = frame.shape[:2]

    # Handle resolution scaling
    if target_width and target_height and (target_width != orig_w or target_height != orig_h):
        scaled_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        scale_x = target_width / float(orig_w)
        scale_y = target_height / float(orig_h)
    else:
        scaled_frame = frame.copy()
        target_width, target_height = orig_w, orig_h
        scale_x, scale_y = 1.0, 1.0

    scale_avg = (scale_x + scale_y) / 2.0

    # Auto-adjust line thicknesses and point radii to match output resolution
    th_box = max(2, int(round(2.5 * scale_avg / 3.0))) if box_thickness is None else int(box_thickness)
    th_line = max(2, int(round(2.5 * scale_avg / 3.0))) if pose_thickness is None else int(pose_thickness)
    rad_joint = max(3, int(round(4.0 * scale_avg / 3.0))) if pose_radius is None else int(pose_radius)
    rad_face = max(2, int(round(2.0 * scale_avg / 3.0))) if face_radius is None else int(face_radius)
    th_hand = max(2, int(round(2.0 * scale_avg / 3.0))) if hand_thickness is None else int(hand_thickness)
    rad_hand = max(3, int(round(4.0 * scale_avg / 3.0))) if hand_radius is None else int(hand_radius)

    canvas = scaled_frame.copy()
    layer = canvas.copy()

    c_pose = hex_to_rgb(pose_color)
    c_face = hex_to_rgb(face_color)
    c_face_det = hex_to_rgb(face_details_color)
    c_face_box = hex_to_rgb(face_box_color)
    c_lh = hex_to_rgb(left_hand_color)
    c_lh_box = hex_to_rgb(left_hand_box_color)
    c_rh = hex_to_rgb(right_hand_color)
    c_rh_box = hex_to_rgb(right_hand_box_color)

    # Helper to scale bounding box
    def scale_box(b):
        if not b:
            return None
        return [
            max(0, int(round(b[0] * scale_x))),
            max(0, int(round(b[1] * scale_y))),
            min(target_width, int(round(b[2] * scale_x))),
            min(target_height, int(round(b[3] * scale_y)))
        ]

    # Helper to scale landmarks list
    def scale_pts(pts):
        if len(pts) == 0:
            return []
        arr = np.array(pts, dtype=float)
        arr[:, 0] *= scale_x
        arr[:, 1] *= scale_y
        return arr

    # 1. Bounding Boxes
    if draw_boxes:
        bf = scale_box(get_bbox(kps, 'bbox_face', 'face_landmarks'))
        if bf and (bf[2] > bf[0]) and (bf[3] > bf[1]):
            cv2.rectangle(layer, (bf[0], bf[1]), (bf[2], bf[3]), c_face_box, th_box, cv2.LINE_AA)

        bl = scale_box(get_bbox(kps, 'bbox_left_hand', 'left_hand_landmarks'))
        if bl and (bl[2] > bl[0]) and (bl[3] > bl[1]):
            cv2.rectangle(layer, (bl[0], bl[1]), (bl[2], bl[3]), c_lh_box, th_box, cv2.LINE_AA)

        br = scale_box(get_bbox(kps, 'bbox_right_hand', 'right_hand_landmarks'))
        if br and (br[2] > br[0]) and (br[3] > br[1]):
            cv2.rectangle(layer, (br[0], br[1]), (br[2], br[3]), c_rh_box, th_box, cv2.LINE_AA)

    # 2. Pose (Torso & Arms)
    p_pts = scale_pts(kps.get('pose_landmarks', []))
    if len(p_pts) > 0 and pose_opacity > 0:
        for p1, p2 in POSE_CONNECTIONS:
            if p1 < len(p_pts) and p2 < len(p_pts):
                pt1 = (int(round(p_pts[p1][0])), int(round(p_pts[p1][1])))
                pt2 = (int(round(p_pts[p2][0])), int(round(p_pts[p2][1])))
                cv2.line(layer, pt1, pt2, c_pose, th_line + 1, cv2.LINE_AA)
        for idx in range(11, min(25, len(p_pts))):
            pt = p_pts[idx]
            x, y = int(round(pt[0])), int(round(pt[1]))
            j_color = COLOR_POSE_GREEN if idx in POSE_GREEN_JOINTS else c_pose
            cv2.circle(layer, (x, y), rad_joint, COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(layer, (x, y), max(1, rad_joint - 1), j_color, -1, cv2.LINE_AA)

    # 3. Face Mesh (Blue base with Green details)
    f_pts = scale_pts(kps.get('face_landmarks', []))
    if len(f_pts) > 0 and face_opacity > 0:
        for idx, pt in enumerate(f_pts):
            x, y = int(round(pt[0])), int(round(pt[1]))
            col = c_face_det if idx in FACE_DETAIL_INDICES else c_face
            cv2.circle(layer, (x, y), rad_face, col, -1, cv2.LINE_AA)

    # 4. Left Hand (Purple skeleton & points)
    lh_pts = scale_pts(kps.get('left_hand_landmarks', []))
    if len(lh_pts) > 0 and left_hand_opacity > 0:
        for p1, p2 in HAND_CONNECTIONS:
            if p1 < len(lh_pts) and p2 < len(lh_pts):
                pt1 = (int(round(lh_pts[p1][0])), int(round(lh_pts[p1][1])))
                pt2 = (int(round(lh_pts[p2][0])), int(round(lh_pts[p2][1])))
                cv2.line(layer, pt1, pt2, c_lh, th_hand, cv2.LINE_AA)
        for pt in lh_pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(layer, (x, y), rad_hand, COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(layer, (x, y), max(1, rad_hand - 1), c_lh, -1, cv2.LINE_AA)

    # 5. Right Hand (Red skeleton & points)
    rh_pts = scale_pts(kps.get('right_hand_landmarks', []))
    if len(rh_pts) > 0 and right_hand_opacity > 0:
        for p1, p2 in HAND_CONNECTIONS:
            if p1 < len(rh_pts) and p2 < len(rh_pts):
                pt1 = (int(round(rh_pts[p1][0])), int(round(rh_pts[p1][1])))
                pt2 = (int(round(rh_pts[p2][0])), int(round(rh_pts[p2][1])))
                cv2.line(layer, pt1, pt2, c_rh, th_hand, cv2.LINE_AA)
        for pt in rh_pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(layer, (x, y), rad_hand, COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(layer, (x, y), max(1, rad_hand - 1), c_rh, -1, cv2.LINE_AA)

    # Overall blend with frame (default 0.88 opacity for crisp, vibrant overlay)
    blend_alpha = float(np.mean([pose_opacity, face_opacity, left_hand_opacity, right_hand_opacity]))
    blend_alpha = max(0.1, min(1.0, blend_alpha))
    if blend_alpha < 1.0:
        canvas = cv2.addWeighted(layer, blend_alpha, canvas, 1.0 - blend_alpha, 0)
    else:
        canvas = layer

    return canvas


# ==============================================================================
# 3. VIDEO RENDERER & HIGH-DEFINITION H.264 ENCODING
# ==============================================================================

def render_keypoints_video(
    video_frames: list,
    keypoints_list: list,
    fps: float = 25.0,
    target_height: int = 1080,
    **render_kwargs
) -> str:
    """
    Draws keypoints across all video frames with automatic 1080p high-definition upscaling
    and encodes them into a crisp, web-playable H.264 MP4 file.

    Args:
        video_frames (list): List of RGB numpy image frames.
        keypoints_list (list): List of dictionary objects containing per-frame landmarks.
        fps (float): Frame rate of the source video.
        target_height (int): Target vertical resolution (default 1080p).
        **render_kwargs: Optional color, opacity, and thickness parameters passed to draw_keypoints_frame.

    Returns:
        str: Absolute path to the generated keypoint overlay video file.
    """
    if not video_frames or not keypoints_list:
        return ""

    try:
        orig_h, orig_w = video_frames[0].shape[:2]
        if fps is None or fps <= 0:
            fps = 25.0

        # Calculate 1080p equivalent dimensions maintaining original aspect ratio
        if orig_h < target_height:
            scale = target_height / float(orig_h)
            out_h = int(target_height)
            out_w = int(round(orig_w * scale))
        else:
            out_h = int(orig_h)
            out_w = int(orig_w)

        # Dimensions must be divisible by 2 for H.264 codec
        out_w = (out_w // 2) * 2
        out_h = (out_h // 2) * 2

        out_dir = os.path.join(tempfile.gettempdir(), "slt_keypoints_videos")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        out_path = os.path.join(out_dir, f"keypoints_1080p_{ts}.mp4")

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",         # Near lossless visual quality
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for frame, kps in zip(video_frames, keypoints_list):
            drawn = draw_keypoints_frame(
                frame=frame,
                kps=kps,
                target_width=out_w,
                target_height=out_h,
                **render_kwargs
            )
            proc.stdin.write(drawn.tobytes())
        proc.stdin.close()
        proc.wait()

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Rendered crisp 1080p keypoints video: {out_w}x{out_h} at {out_path}")
            return out_path
    except Exception as e:
        print(f"Error rendering keypoints video: {e}")

    return ""
