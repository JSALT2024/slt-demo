import os
import sys
import time
import torch
import numpy as np
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def format_log(text: str, start_time: float = None, target_col: int = 64) -> str:
    """Formats a log message with tabulators aligning timestamps in a clean column."""
    if start_time is not None:
        elapsed = time.time() - start_time
        time_str = f"[{elapsed:6.2f} s since start]"
    else:
        time_str = ""
    num_tabs = max(1, (target_col - len(text) + 7) // 8)
    tabs = "\t" * num_tabs
    return f"{text}{tabs}{time_str}"

# Import the translation model and helper functions from Uni_Sign
from Uni_Sign.models import Uni_Sign
from Uni_Sign.datasets import load_part_kp_YTASL, YTASL_GROUP_SIZES, _fill_missing_landmarks, select_frame_indices
from predict_pose import create_mediapipe_models, predict_pose, load_video_cv
from visualize_pose import render_keypoints_video

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

# Global variables to cache models and config so they are not reloaded on every video translation
model = None
pose_models = None
args = None

class InferenceConfig:
    """Configuration class for Uni_Sign model inference."""
    def __init__(self):
        # Path to the pre-trained weights checkpoint
        self.finetune = os.environ.get("UNISIGN_WEIGHTS", r"./Uni_Sign/unisign_model/best_checkpoint-wlasl.pth")
        self.dataset = "YTASL"
        self.task = "SLT"
        self.max_length = 256
        self.normalization = "none"
        self.layout = "pruned"
        self.n_registers = 0
        self.hidden_dim = 256
        self.rgb_support = False
        self.no_adaptive_gcn = False
        self.register_position = "before_all"
        self.label_smoothing = 0.2
        self.batch_size = 1

def initialize_model():
    """Initializes the Uni_Sign translator and the Mediapipe/YOLO pose estimation models."""
    global model, pose_models, args
    
    if model is not None and pose_models is not None:
        return
    
    print("Initializing Uni_Sign model...")
    args = InferenceConfig()
    model = Uni_Sign(args=args)
    
    # Load model weights
    if args.finetune and os.path.exists(args.finetune):
        print(f"Loading checkpoint weights from: {args.finetune}")
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: Checkpoint weights not found at '{args.finetune}'!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Initializing Mediapipe and YOLO models...")
    pose_checkpoint_folder = 'checkpoints/pose/'
    pose_models = create_mediapipe_models(pose_checkpoint_folder)

def process_pose_data_in_memory(pose_results, args):
    """
    Processes pose keypoints in-memory from dictionary results.
    Converts keypoint data to Tensors formatted for Uni_Sign input.
    """
    raw_pose = pose_results.get('cropped_keypoints', [])
    if not raw_pose:
        raise ValueError("Video does not contain any detected pose (cropped_keypoints).")

    # Format raw coordinates into standard [X, Y] coordinates for each body part
    pose = []
    for frame_data in raw_pose:
        formatted_frame = {}
        for part, expected_size in YTASL_GROUP_SIZES.items():
            kps = frame_data.get(part, [])
            
            # Use empty list if data is missing; _fill_missing_landmarks will fill it
            if kps is None or len(kps) == 0:
                 formatted_frame[part] = [] 
            else:
                 # Standardize to only X and Y coordinates (ignoring confidence/Z columns)
                 formatted_frame[part] = np.array(kps)[:, :2].tolist()
        pose.append(formatted_frame)

    # Subsample frames to match target length (max_length)
    duration = len(pose)
    tmp = select_frame_indices(duration, args.max_length, phase='test')
    skeletons = [pose[i] for i in tmp]

    # Fill in missing landmarks to maintain consistent shape dimensions
    confs = []
    for i, skeleton in enumerate(skeletons):
        conf = {}
        for group_name, expected_size in YTASL_GROUP_SIZES.items():
            _fill_missing_landmarks(
                skeleton=skeleton,
                conf=conf,
                group_name=group_name,
                expected_size=expected_size,
                clip_name="gradio_video",
                frame_idx=i,
                error_group_label=f"group '{group_name}'",
                include_size_details=False,
                strict_key_access=True,
            )
        confs.append(conf)

    # Normalize keypoints and structure them for YTASL dataset
    kps_with_scores = load_part_kp_YTASL(skeletons, confs, args.normalization, args.layout)

    # Batch features by unsqueezing to add batch dimension (batch_size = 1)
    src_input = {}
    for key, val in kps_with_scores.items():
        src_input[key] = val.unsqueeze(0)

    # Generate attention mask and sequence lengths
    seq_len = src_input['body'].shape[1]
    mask_gen = torch.ones([seq_len]) + 7
    src_input['attention_mask'] = (mask_gen != 0).long().unsqueeze(0)
    src_input['src_length_batch'] = torch.LongTensor([seq_len])
    src_input['name_batch'] = ["gradio_video"]

    return src_input


def process_input(input_video_path, progress=None):
    """
    Main entry point for Gradio. Runs pose extraction, pre-processing, 
    and model translation inference via Uni_Sign.
    """
    try:
        start_time = time.time()
        
        print(format_log("0. Initializing models...", start_time))
        if progress is not None:
            try:
                progress(0.02, desc="0. Inicializace modelů...")
            except Exception:
                pass
        initialize_model()
        
        print(format_log("1. Extracting keypoints from video...", start_time))
        if progress is not None:
            try:
                progress(0.05, desc="[1a] Načítání videa...")
            except Exception:
                pass
        video_frames, fps = load_video_cv(input_video_path)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 25.0

        num_frames = len(video_frames)
        dur = num_frames / fps if fps > 0 else 0
        print(format_log(f"   [1a] Video načteno: {num_frames} snímků ({fps:.1f} FPS, délka {dur:.1f} s)", start_time))

        pose_results = predict_pose(video_frames, pose_models, progress=progress, start_time=start_time)

        # Attach bounding boxes to each frame's keypoints
        kps_list = pose_results.get("keypoints", [])
        bbox_face_list = pose_results.get("bbox_face", [])
        bbox_lh_list = pose_results.get("bbox_left_hand", [])
        bbox_rh_list = pose_results.get("bbox_right_hand", [])

        for i, kp in enumerate(kps_list):
            if i < len(bbox_face_list):
                kp['bbox_face'] = bbox_face_list[i]
            if i < len(bbox_lh_list):
                kp['bbox_left_hand'] = bbox_lh_list[i]
            if i < len(bbox_rh_list):
                kp['bbox_right_hand'] = bbox_rh_list[i]

        # Render keypoint overlay video
        keypoints_video_path = ""
        try:
            keypoints_video_path = render_keypoints_video(video_frames, kps_list, fps=fps, progress=progress, start_time=start_time)
            print(format_log("   [1e] Video s keypointy úspěšně uloženo", start_time))
        except Exception as kp_err:
            print(format_log(f"   [1e] Warning rendering keypoints video: {kp_err}", start_time))
        
        print(format_log("2. Pre-processing visual features...", start_time))
        if progress is not None:
            try:
                progress(0.91, desc="2. Zpracování vizuálních příznaků...")
            except Exception:
                pass

        print(format_log("3. Converting keypoints to Tensors for Uni_Sign...", start_time))
        if progress is not None:
            try:
                progress(0.93, desc="3. Příprava tenzorů pro model...")
            except Exception:
                pass
        src_input = process_pose_data_in_memory(pose_results, args)
        
        # Move Tensors to the same device (GPU/CPU) as the model
        device = next(model.parameters()).device
        for key in ['body', 'left', 'right', 'face_all', 'attention_mask']:
            if key in src_input:
                # Cast features to float32 (except the attention mask)
                if key != 'attention_mask':
                    src_input[key] = src_input[key].to(device, dtype=torch.float32)
                else:
                    src_input[key] = src_input[key].to(device)
                    
        tgt_input = {'gt_sentence': [""], 'gt_gloss': [""]}
        
        print(format_log("4. Generating translation...", start_time))
        if progress is not None:
            try:
                progress(0.96, desc="4. Generování překladu...")
            except Exception:
                pass
        with torch.no_grad():
            stack_out = model(src_input, tgt_input)
            output = model.generate(
                stack_out,
                max_new_tokens=100,
                num_beams=4,
            )
            
        tokenizer = model.mt5_tokenizer
        tgt_pres = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        result = tgt_pres[0].strip()
        if not result:
            result = "The model was unable to generate a translation. Please try a different video."
        
        if progress is not None:
            try:
                progress(1.0, desc="5. Překlad dokončen!")
            except Exception:
                pass
        print(format_log("5. Translation completed!", start_time))
        return result, keypoints_video_path
                
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing video: {str(e)}", ""
