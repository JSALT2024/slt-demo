# Sign Language Translation (SLT) Demo

This project provides an end-to-end Sign Language Translation system that translates American Sign Language (ASL) videos into English text. 

The system extracts spatial-temporal skeleton keypoints from video frames and processes them through the **Uni_Sign** deep learning model.

---

## 🛠️ System Architecture

The pipeline consists of three core components:

1. **Frontend Web UI ([app.py](app.py))**:
   - A web-based interface built with **Gradio**.
   - Allows users to upload an ASL video, initiate translation, and view the text translation.
   - Automatically downloads the trained model weights (~1GB) from Hugging Face Hub on first launch.

2. **Keypoints Extraction ([predict_pose.py](predict_pose.py))**:
   - Detects the signing person using a **YOLOv8-pose** model to isolate the "signing space".
   - Runs **MediaPipe** (Pose, Face, and Hand landmarker tasks) on the cropped signing space to extract coordinates.
   - Maps left vs. right hands dynamically based on wrist coordinates.
   - Saves cropped regional frames (face, hands) for feature extraction.

3. **Backend Translation Pipeline ([backend.py](backend.py))**:
   - Restructures extracted keypoints to fit the YouTube-ASL dataset format.
   - Runs model inference using **Uni_Sign** (which processes body, hands, and face coordinates with Spatio-Temporal Graph Convolutional Networks (STGCN) and deformable attention).
   - Generates English text translations using an MT5 tokenizer.

---

## 🔄 Execution Pipeline & Responsibilities

This step-by-step list details how data flows through the application and which functions handle each step:

1. **Model Cache & Initialization**: 
   - `app.py` checks and downloads checkpoint weights from Hugging Face.
   - `backend.py` $\rightarrow$ `initialize_model()`: Caches and initializes models and configurations, loading Uni_Sign weights onto target hardware (GPU/CPU).
2. **Video Frame Reading**:
   - `predict_pose.py` $\rightarrow$ `load_video_cv()`: Reads all frames of the input video and converts them to RGB format.
3. **Primary Subject Localization**:
   - `predict_pose.py` $\rightarrow$ `yolo_predict()`: Runs YOLOv8-pose to locate human body bounding boxes.
   - `predict_pose.py` $\rightarrow$ `new_bbox()`: Calculates the frame's signing space based on shoulder distance. The median box of all frames is chosen to avoid camera jitter.
4. **Detailed Landmark Extraction**:
   - `predict_pose.py` $\rightarrow$ `predict_pose()`: Runs MediaPipe Pose, Hands, and Face detectors on the cropped YOLO region.
   - `predict_pose.py` $\rightarrow$ `keypoints_out_format()` & `mediapipe_to_xy()`: Convert coordinate formats.
   - `predict_pose.py` $\rightarrow$ `process_hands()`: Resolves left/right hand coordinate associations via Hungarian matching (`linear_sum_assignment`).
5. **Secondary Cropping and Dino Crops**:
   - `predict_pose.py` $\rightarrow$ `crop_pad_image()`: Crops the video to a static squared MediaPipe median bounding box.
   - `predict_pose.py` $\rightarrow$ `get_centered_box()` & `adjust_bounding_box()`: Generate localized region crops (Dino crops) for face, left hand, and right hand.
6. **Keypoint Normalization & Formatting**:
   - `backend.py` $\rightarrow$ `process_pose_data_in_memory()`: Standardizes coordinates to 2D lists, pads missing frames via `_fill_missing_landmarks()`, and normalizes them into Tensors using `load_part_kp_YTASL()`.
7. **Model Translation Inference**:
   - `backend.py` $\rightarrow$ `process_input()`: Standardizes features, transfers Tensors to target hardware device, runs the model forward pass, and decodes the translation output using `model.generate()`.