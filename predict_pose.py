import os
from copy import deepcopy
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def crop_frame(image: np.ndarray, bounding_box: tuple) -> np.ndarray:
    """Crops a rectangular region from an image based on the provided bounding box."""
    x, y, w, h = bounding_box
    return image[y:y + h, x:x + w]


def get_centered_box(keypoints: np.ndarray, box_size: int, scale_factor: float = 1.2) -> tuple:
    """Calculates a square bounding box centered around keypoints with scaling padding."""
    center_x, center_y = np.mean(keypoints, axis=0, dtype=int)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size

    # Add scaling padding to avoid tight crop cuts
    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding

    return x, y, w, h


def adjust_bounding_box(bounding_box: tuple, image_shape: tuple) -> tuple:
    """Adjusts the bounding box coordinates to ensure they remain within image boundaries."""
    x, y, w, h = bounding_box
    ih, iw, _ = image_shape

    # Prevent box from exceeding the right boundary
    if x + w > iw:
        x = iw - w

    # Prevent box from exceeding the bottom boundary
    if y + h > ih:
        y = ih - h

    # Prevent negative coordinate offsets
    x = max(x, 0)
    y = max(y, 0)

    return x, y, w, h


def create_mediapipe_models(checkpoint_folder: str, min_confidence: float = 0.4) -> tuple:
    """Initializes the MediaPipe (Pose, Hands, Face) and YOLOv8 models."""
    BaseOptions = mp.tasks.BaseOptions

    # Define model asset paths
    hand_model_path = os.path.join(checkpoint_folder, 'hand_landmarker.task')
    pose_model_path = os.path.join(checkpoint_folder, 'pose_landmarker_full.task')
    face_model_path = os.path.join(checkpoint_folder, 'face_landmarker.task')
    yolo_model_path = os.path.join(checkpoint_folder, "yolov8n-pose.pt")

    # Load YOLOv8-pose model
    yolo_model = YOLO(yolo_model_path)

    # Configure and create MediaPipe Hand landmarker
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        num_hands=2
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # Configure and create MediaPipe Pose landmarker
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        num_poses=1
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    # Configure and create MediaPipe Face landmarker
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        num_faces=1
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    return hand_detector, pose_detector, face_detector, yolo_model


def yolo_predict(image: np.ndarray, model, min_conf: float = 0.5) -> tuple:
    """Runs YOLOv8-pose on an image and filters predictions by confidence threshold."""
    yolo_results = model(image, verbose=False)

    bboxes = yolo_results[0].boxes.xyxy
    keypoints = yolo_results[0].keypoints.xy
    bboxes = bboxes.cpu().numpy()
    keypoints = keypoints.cpu().numpy()

    conf = yolo_results[0].boxes.conf
    conf = conf.cpu().numpy()
    
    # Filter out empty detections and low-confidence boundaries
    select_mask_kp = np.sum(keypoints, axis=(1, 2)) > 0.0001
    select_mask_bb = conf > min_conf
    select_mask = select_mask_kp & select_mask_bb

    conf = conf[select_mask]
    bboxes = bboxes[select_mask]
    keypoints = keypoints[select_mask]

    return bboxes, keypoints, conf


def load_video_cv(path: str) -> tuple:
    """Loads all frames from a video file path and converts them to RGB format."""
    video = []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            # Convert default BGR frame to RGB for processing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
    cap.release()
    return video, fps


def new_bbox(image: np.ndarray, keypoints: np.ndarray, lsi: int = 5, rsi: int = 6, sign_space: float = 5.0) -> tuple:
    """Calculates a signing bounding box based on shoulder coordinates and keypoint extent."""
    h, w = image.shape[:2]
    l_shoulder = keypoints[lsi]
    r_shoulder = keypoints[rsi]
    distance = np.sqrt((l_shoulder[0] - r_shoulder[0]) ** 2 + (l_shoulder[1] - r_shoulder[1]) ** 2)

    # Determine mid-point center between shoulders
    center_x = np.abs(l_shoulder[0] - r_shoulder[0]) / 2 + np.min([l_shoulder[0], r_shoulder[0]], 0)
    center_y = np.abs(l_shoulder[1] - r_shoulder[1]) / 2 + np.min([l_shoulder[1], r_shoulder[1]], 0)

    # Define bounding box limits based on shoulder distance scaling
    new_x0 = center_x - (distance * (sign_space / 2))
    new_x1 = center_x + (distance * (sign_space / 2))
    new_y0 = center_y - (distance * (sign_space / 2))
    new_y1 = center_y + (distance * (sign_space / 2))

    # Expand bounding box coordinates to include all active keypoints
    idx_x = keypoints[:, 0] > 0
    idx_y = keypoints[:, 1] > 0
    new_x0 = np.min([new_x0, *keypoints[idx_x, 0]])
    new_x1 = np.max([new_x1, *keypoints[idx_x, 0]])
    new_y0 = np.min([new_y0, *keypoints[idx_y, 1]])
    new_y1 = np.max([new_y1, *keypoints[idx_y, 1]])

    # Bound and crop coordinates to image shape
    new_x0 = np.round(np.clip(new_x0, 0, w)).astype(int)
    new_x1 = np.round(np.clip(new_x1, 0, w)).astype(int)
    new_y0 = np.round(np.clip(new_y0, 0, h)).astype(int)
    new_y1 = np.round(np.clip(new_y1, 0, h)).astype(int)

    return new_x0, new_y0, new_x1, new_y1


def mediapipe_to_xy(data, image_size: tuple = None) -> tuple:
    """Extracts absolute [x, y] coordinates from MediaPipe normalized landmark objects."""
    x = np.array([kp.x for kp in data])
    y = np.array([kp.y for kp in data])

    if image_size is not None:
        x = x * image_size[1]
        y = y * image_size[0]

    return x, y


def crop_pad_image(image: np.ndarray, bbox: np.ndarray, border: float = 0.25) -> tuple:
    """Crops the image to a bounding box, pads it to make it a square, and adds a surrounding border."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0

    # Calculate padding offsets to make the region a square
    dif = np.abs(w - h)
    pad_value_0 = np.floor(dif / 2).astype(int)
    pad_value_1 = dif - pad_value_0

    if w > h:
        y0 -= pad_value_0
        y1 += pad_value_1
    else:
        x0 -= pad_value_0
        x1 += pad_value_1

    # Add extra border padding around the squared region
    border = np.round((np.max([w, h]) * border) / 2).astype(int)
    ih, iw = image.shape[:2]
    y0 -= border
    y1 += border
    x0 -= border
    x1 += border

    new_bbox = [x0, y0, x1, y1]

    # Convert coordinates to absolute offsets on padded canvas
    y0 += ih
    y1 += ih
    x0 += iw
    x1 += iw

    # Pad image to prevent out-of-bounds cropping
    image = np.pad(image, ((ih, ih), (iw, iw), (0, 0)), mode='constant', constant_values=0)
    cropped_image = image[y0:y1, x0:x1]

    return cropped_image, new_bbox


def keypoints_out_format(mp_keypoints, image_size: tuple) -> np.ndarray:
    """Formats MediaPipe landmark objects into a standard numpy array containing [x, y, z, visibility]."""
    if len(mp_keypoints) >= 1:
        data = mp_keypoints[0]
        x, y = mediapipe_to_xy(data, image_size)
        z = np.array([kp.z for kp in data])
        visibility = np.array([kp.visibility for kp in data])
        data = np.array([x, y, z, visibility]).T
        return data
    else:
        return []


def distance_matrix(P: list, Q: list) -> np.ndarray:
    """Computes a Euclidean distance matrix between two lists of coordinates."""
    dis_max = np.zeros([len(P), len(Q)])
    for i, p in enumerate(P):
        for j, q in enumerate(Q):
            dist = np.linalg.norm(np.array(p) - np.array(q))
            dis_max[i, j] = dist
    return dis_max


def process_hands(mp_hand_keypoints, mp_handedness, pose_keypoints, image_size: tuple, yolo_pose_keypoints=None) -> dict:
    """
    Tracks and assigns detected hands to left/right sides.
    Uses Hungarian matching based on wrist coordinate distances if both hands are detected.
    """
    out = {"left": [], "right": []}

    if len(mp_hand_keypoints) == 0:
        return out

    # Format MediaPipe hand coordinates
    hand_keypoints = []
    for data in mp_hand_keypoints:
        hand_keypoints.append(keypoints_out_format([data], image_size))

    # Fast assignment if only a single hand is detected
    if len(mp_hand_keypoints) == 1:
        # mp_handedness[0] is a list of Category objects for the detected hand
        side = mp_handedness[0][0].category_name.lower()
        out[side] = hand_keypoints[0]
        return out

    # Compute centers for each detected hand region
    hand_centers = []
    for keypoints in hand_keypoints:
        x = keypoints[0, 0]
        y = keypoints[0, 1]
        hand_center = [x, y]
        hand_centers.append(hand_center)

    # Locate wrist coordinates from Pose/YOLO estimations
    left_wrist = None
    right_wrist = None

    pose_keypoints = None if len(pose_keypoints) == 0 else pose_keypoints
    if pose_keypoints is not None:
        left_wrist = pose_keypoints[15, :2]
        right_wrist = pose_keypoints[16, :2]
    elif pose_keypoints is None and yolo_pose_keypoints is not None:
        left_wrist = yolo_pose_keypoints[9, :2]
        right_wrist = yolo_pose_keypoints[10, :2]
        if (np.sum(left_wrist) == 0) or (np.sum(right_wrist) == 0):
            left_wrist = None
            right_wrist = None

    # Perform assignment based on proximity to wrist locations
    if left_wrist is not None and right_wrist is not None:
        wrists = [left_wrist, right_wrist]

        dis_max = distance_matrix(wrists, hand_centers)
        row_idx, col_idx = linear_sum_assignment(dis_max)

        sides = list(out.keys())
        for ridx, cidx in zip(row_idx, col_idx):
            side = sides[ridx]
            keypoints = hand_keypoints[cidx]
            out[side] = keypoints
    else:
        # Fallback to simple spatial heuristic (left hand on the right side of the frame, etc.)
        hand_centers_x = np.array(hand_centers)[:, 0]
        right_idx = np.argmin(hand_centers_x)
        out["right"] = hand_keypoints[right_idx]
        left_idx = np.argmax(hand_centers_x)
        if right_idx != left_idx:
            out["left"] = hand_keypoints[left_idx]

    return out


def predict_pose(video: List[np.ndarray], models: tuple, sign_space=4, yolo_sign_space=4) -> dict:
    """
    Extracts keypoints and cropped regional structures from video frames.
    
    1. Runs YOLOv8-pose to compute the subject's signing box center.
    2. Runs MediaPipe Pose, Hands, and Face landmarkers on cropped YOLO space.
    3. Aligns and shifts all landmarks into standard coordinate frames.
    4. Crops face and hand boxes (DINO crops) for visual translation features.
    """
    hand_detector, pose_detector, face_detector, yolo_model = models
    results = {
        "images": video,
        "keypoints": [],
        "cropped_images": [],
        "cropped_keypoints": [],
        "sign_space": [],
        "cropped_left_hand": [],
        "cropped_right_hand": [],
        "cropped_face": [],
        "bbox_left_hand": [],
        "bbox_right_hand": [],
        "bbox_face": [],
    }

    # Step 1: Run YOLO detector across frames to determine primary human center
    yolo_predictions = []
    num_predictions = []
    for idx, image in enumerate(results["images"]):
        bboxes, keypoints, confs = yolo_predict(image, yolo_model)
        yolo_predictions.append([bboxes, keypoints, confs])
        num_predictions.append(len(bboxes))
        
    # Return zeroed outputs if no human subject is detected in any frames
    if np.sum(num_predictions) == 0:
        _h, _w = results["images"][0].shape[:2]
        for idx in range(len(results["images"])):
            results["keypoints"].append({'pose_landmarks': [], 'right_hand_landmarks': [], 'left_hand_landmarks': [], 'face_landmarks': []})
            results["cropped_images"].append(results["images"][idx])
            results["cropped_keypoints"].append({'pose_landmarks': [], 'right_hand_landmarks': [], 'left_hand_landmarks': [], 'face_landmarks': []})
            results["sign_space"].append([0, 0, _w, _h])
            results["cropped_left_hand"].append(np.zeros([224, 224, 3], dtype=np.uint8))
            results["cropped_right_hand"].append(np.zeros([224, 224, 3], dtype=np.uint8))
            results["cropped_face"].append(np.zeros([224, 224, 3], dtype=np.uint8))
            results["bbox_left_hand"].append([])
            results["bbox_right_hand"].append([])
            results["bbox_face"].append([])    
        return results

    # Calculate median YOLO bounding box across the video segment
    x0, y0, x1, y1 = [], [], [], []
    for idx, (image, prediction) in enumerate(zip(results["images"], yolo_predictions)):
        _, keypoints, _ = prediction
        if len(keypoints) != 1:
            continue

        _x0, _y0, _x1, _y1 = new_bbox(image, keypoints[0], lsi=5, rsi=6, sign_space=yolo_sign_space)

        x0.append(_x0)
        y0.append(_y0)
        x1.append(_x1)
        y1.append(_y1)

    x0y = np.round(np.median(x0)).astype(int)
    y0y = np.round(np.median(y0)).astype(int)
    x1y = np.round(np.median(x1)).astype(int)
    y1y = np.round(np.median(y1)).astype(int)

    # Step 2: Extract fine-grained landmarks using MediaPipe inside the YOLO crop region
    mp_keypoints_list = []
    x0, y0, x1, y1 = [], [], [], []
    for idx, image in enumerate(results["images"]):
        yolo_image = image[y0y:y1y, x0y:x1y]
        yih, yiw = yolo_image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(yolo_image))

        # Fallback empty canvas if crop goes out-of-bounds
        if yolo_image.shape == (0, 0, 3):
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.zeros(shape=(256, 256, 3), dtype=np.uint8)
            )

        pose_prediction = pose_detector.detect(mp_image)
        hand_prediction = hand_detector.detect(mp_image)
        face_prediction = face_detector.detect(mp_image)

        face_keypoints = keypoints_out_format(face_prediction.face_landmarks, (yih, yiw))
        pose_keypoints = keypoints_out_format(pose_prediction.pose_landmarks, (yih, yiw))
        hand_keypoints = process_hands(
            hand_prediction.hand_landmarks,
            hand_prediction.handedness,
            pose_keypoints,
            (yih, yiw),
            None
        )

        keypoints = {
            'pose_landmarks': pose_keypoints,
            'right_hand_landmarks': hand_keypoints["right"],
            'left_hand_landmarks': hand_keypoints["left"],
            'face_landmarks': face_keypoints
        }

        # Shift keypoints to original image coordinates by adding YOLO offsets
        for name in keypoints:
            if len(keypoints[name]) > 0:
                keypoints[name] = np.array(keypoints[name], dtype=float)
                keypoints[name][:, 0] += x0y
                keypoints[name][:, 1] += y0y

        # Gather points to determine frame's specific signing bounding box
        kp_all_x = []
        kp_all_y = []
        
        # Use only first 25 pose landmarks (torso and shoulders) to define bounds
        if len(keypoints['pose_landmarks']) > 0:
            kp_all_x.extend(keypoints['pose_landmarks'][:25, 0])
            kp_all_y.extend(keypoints['pose_landmarks'][:25, 1])
        if len(keypoints['face_landmarks']) > 0:
            kp_all_x.extend(keypoints['face_landmarks'][:, 0])
            kp_all_y.extend(keypoints['face_landmarks'][:, 1])
        if len(keypoints['left_hand_landmarks']) > 0:
            kp_all_x.extend(keypoints['left_hand_landmarks'][:, 0])
            kp_all_y.extend(keypoints['left_hand_landmarks'][:, 1])
        if len(keypoints['right_hand_landmarks']) > 0:
            kp_all_x.extend(keypoints['right_hand_landmarks'][:, 0])
            kp_all_y.extend(keypoints['right_hand_landmarks'][:, 1])

        if len(kp_all_x) > 0:
            kp_all = np.array((kp_all_x, kp_all_y)).T
            _x0, _y0, _x1, _y1 = new_bbox(image, kp_all, lsi=11, rsi=12, sign_space=sign_space)
            x0.append(_x0)
            y0.append(_y0)
            x1.append(_x1)
            y1.append(_y1)

        mp_keypoints_list.append(keypoints)

    # Compute a median MediaPipe signing box to prevent coordinate jitter
    if len(x0) == 0:
        ih, iw = video[0].shape[:2]
        x0mp = 0
        y0mp = 0
        x1mp = iw
        y1mp = ih
    else:
        x0mp = np.round(np.median(x0)).astype(int)
        y0mp = np.round(np.median(y0)).astype(int)
        x1mp = np.round(np.median(x1)).astype(int)
        y1mp = np.round(np.median(y1)).astype(int)

    # Step 3: Crop and extract final features (Loop 3)
    for idx, (image, keypoints) in enumerate(zip(results["images"], mp_keypoints_list)):
        cropped_image, pad_bbox = crop_pad_image(image, (x0mp, y0mp, x1mp, y1mp), border=0)

        # Generate Dino crops for hands and face regions
        name_to_keypoints = [
            ("face", keypoints['face_landmarks']),
            ("left_hand", keypoints['left_hand_landmarks']),
            ("right_hand", keypoints['right_hand_landmarks'])
        ]
        for name, kp in name_to_keypoints:
            if len(kp) > 0:
                kp = np.array(kp, dtype=float)
                kp = np.round(kp[:, :2]).astype(int)
                x, y, w, h = cv2.boundingRect(kp)
                cropped_local_bbox = get_centered_box(kp, np.max([w, h]), scale_factor=1.2)
                cropped_local_bbox = adjust_bounding_box(cropped_local_bbox, image.shape)
                cropped_local_image = crop_frame(image, cropped_local_bbox)
                x0_box, y0_box, w_box, h_box = cropped_local_bbox
                cropped_local_bbox = [x0_box, y0_box, x0_box + w_box, y0_box + h_box]
            else:
                cropped_local_image = np.zeros([224, 224, 3], dtype=np.uint8)
                cropped_local_bbox = []
            results[f"bbox_{name}"].append(cropped_local_bbox)
            results[f"cropped_{name}"].append(cropped_local_image)

        # Offset keypoints to final cropped coordinates
        x_move = pad_bbox[0]
        y_move = pad_bbox[1]
        keypoints_cropped = deepcopy(keypoints)
        for name in keypoints_cropped:
            if len(keypoints_cropped[name]) > 0:
                keypoints_cropped[name][:, 0] -= x_move
                keypoints_cropped[name][:, 1] -= y_move
                
                # Slice to 2D coordinates [X, Y] and round to 3 decimal places
                keypoints_cropped[name] = np.round(keypoints_cropped[name][:, :2], 3).tolist()
                keypoints[name] = np.round(keypoints[name][:, :2], 3).tolist()

        # Cache processing outputs
        results["keypoints"].append(keypoints)
        results["cropped_images"].append(cropped_image)
        results["cropped_keypoints"].append(keypoints_cropped)
        results["sign_space"].append(pad_bbox)
    results["images"] = video

    return results