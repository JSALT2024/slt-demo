import os
import time
import torch
import numpy as np
from dotenv import load_dotenv

# Import the translation model and helper functions from Uni_Sign
from Uni_Sign.models import Uni_Sign
from Uni_Sign.datasets import load_part_kp_YTASL, YTASL_GROUP_SIZES, _fill_missing_landmarks, select_frame_indices
from predict_pose import create_mediapipe_models, predict_pose, load_video_cv

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
        self.finetune = os.environ.get("UNISIGN_WEIGHTS", r"./Uni_Sign/unisign_model/best_checkpoint.pth")
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


def process_input(input_video_path):
    """
    Main entry point for Gradio. Runs pose extraction, pre-processing, 
    and model translation inference via Uni_Sign.
    """
    try:
        start_time = time.time()
        
        print("0. Initializing models...")
        initialize_model()
        
        print(f"1. Extracting keypoints from video... [{time.time() - start_time:.2f} s since start]")
        video_frames, _ = load_video_cv(input_video_path)
        pose_results = predict_pose(video_frames, pose_models)
        
        print(f"2. Pre-processing visual features... [{time.time() - start_time:.2f} s since start]")

        print(f"3. Converting keypoints to Tensors for Uni_Sign... [{time.time() - start_time:.2f} s since start]")
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
        
        print(f"4. Generating translation... [{time.time() - start_time:.2f} s since start]")
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
        
        print(f"5. Translation completed! [{time.time() - start_time:.2f} s since start]")
        return result
                
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing video: {str(e)}"
