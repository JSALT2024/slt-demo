import os
import sys
import torch
import yaml
from transformers import T5Tokenizer
from model.configuration_t5 import SignT5Config
from model.modeling_t5 import T5ModelForSLT
from utils.translation import postprocess_text
import numpy as np
from dotenv import load_dotenv
import cv2

# Import pose extraction functions from predict_pose.py
from predict_pose import (
    create_mediapipe_models, 
    predict_pose, 
    load_video_cv
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()
# set KMP_DUPLICATE_LIB_OK=TRUE

# Initialize global variables for model, tokenizer, and config
model = None
tokenizer = None
config = None
pose_models = None

def load_config(cfg_path='configs/predict_config_demo.yaml'):
    """Load config from a yaml file."""
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    for param, value in cfg['EvaluationArguments'].items():
        if value == 'none' or value == 'None':
            cfg['EvaluationArguments'][param] = None
    return cfg

def get_sign_input_dim(config):
    """Calculate the sign input dimension based on enabled features."""
    sign_input_dim = 0
    for mod in config['SignDataArguments']['visual_features']:
        if config['SignDataArguments']['visual_features'][mod]['enable_input']:
            sign_input_dim += config['SignModelArguments']['projectors'][mod]['dim']
    return sign_input_dim

def initialize_model():
    """Initialize the model and tokenizer."""
    global model, tokenizer, config, pose_models
    
    if model is not None and tokenizer is not None and pose_models is not None:
        return
    
    # Load configuration
    config = load_config()
    evaluation_config = config['EvaluationArguments']
    model_config = config['ModelArguments']
    model_config['sign_input_dim'] = get_sign_input_dim(config)

    # Initialize model configuration
    t5_config = SignT5Config()
    for param, value in model_config.items():
        if hasattr(t5_config, param):
            setattr(t5_config, param, value)

    # Load model and tokenizer
    model = T5ModelForSLT.from_pretrained(evaluation_config['model_dir'], config=t5_config)
    model.config.output_attentions = True
    for param in model.parameters():
        param.data = param.data.contiguous()
    
    tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name, clean_up_tokenization_spaces=True)
    
    # Initialize pose models
    pose_checkpoint_folder = 'checkpoints/pose/'
    pose_models = create_mediapipe_models(pose_checkpoint_folder)
    
    # Move model to appropriate device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

def flatten_keypoints(keypoints_dict):
    """
    Flatten keypoints from all body parts into a single feature vector.
    
    Args:
        keypoints_dict: Dict with pose, hand, and face landmarks
        
    Returns:
        Flattened array of keypoints
    """
    flattened = []
    
    # Process pose keypoints (33 keypoints × 4 values)
    if len(keypoints_dict['pose_landmarks']) > 0:
        pose_kp = np.array(keypoints_dict['pose_landmarks'])
        flattened.extend(pose_kp.flatten())
    else:
        flattened.extend(np.zeros(33 * 4))
    
    # Process right hand keypoints (21 keypoints × 4 values)
    if len(keypoints_dict['right_hand_landmarks']) > 0:
        right_hand_kp = np.array(keypoints_dict['right_hand_landmarks'])
        flattened.extend(right_hand_kp.flatten())
    else:
        flattened.extend(np.zeros(21 * 4))
    
    # Process left hand keypoints (21 keypoints × 4 values)
    if len(keypoints_dict['left_hand_landmarks']) > 0:
        left_hand_kp = np.array(keypoints_dict['left_hand_landmarks'])
        flattened.extend(left_hand_kp.flatten())
    else:
        flattened.extend(np.zeros(21 * 4))
    
    return np.array(flattened)

def extract_pose_features(video_path):
    """
    Extract pose features from the input video using MediaPipe.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Tensor of pose features and attention mask
    """
    global pose_models
    
    # Ensure pose models are initialized
    if pose_models is None:
        initialize_model()
    
    # Load video frames
    video_frames, _ = load_video_cv(video_path)
    
    # Get pose predictions
    pose_results = predict_pose(video_frames, pose_models)
    
    # Extract keypoints for each frame
    max_sequence_length = config['EvaluationArguments']['max_sequence_length']
    pose_dim = config['SignModelArguments']['projectors']['pose']['dim']
    
    # Initialize features tensor with zeros
    features = torch.zeros(max_sequence_length, pose_dim)
    attention_mask = torch.zeros(max_sequence_length)
    
    num_frames = min(len(pose_results['cropped_keypoints']), max_sequence_length)
    
    for i in range(num_frames):
        keypoints = pose_results['cropped_keypoints'][i]
        flat_keypoints = flatten_keypoints(keypoints)
        
        # Ensure the features have the correct dimensions
        if len(flat_keypoints) > pose_dim:
            flat_keypoints = flat_keypoints[:pose_dim]
        elif len(flat_keypoints) < pose_dim:
            flat_keypoints = np.pad(flat_keypoints, (0, pose_dim - len(flat_keypoints)))
            
        features[i] = torch.tensor(flat_keypoints, dtype=torch.float32)
        attention_mask[i] = 1.0  # Mark this frame as valid
    
    return features, attention_mask

def process_input(input_video_path):
    """
    Process a single video input and return the translated text.
    
    Args:
        input_video_path: Path to the input video file
        
    Returns:
        str: The translated text
    """
    # Initialize model if not already done
    initialize_model()
    
    # Extract pose features from the video
    features, attention_mask = extract_pose_features(input_video_path)
    
    # Prepare input batch
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    batch = {
        "sign_inputs": features.unsqueeze(0).to(device).to(model_dtype),
        "attention_mask": attention_mask.unsqueeze(0).to(device).to(model_dtype),
        # No labels needed for inference
    }
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            early_stopping=model.config.early_stopping,
            no_repeat_ngram_size=model.config.no_repeat_ngram_size,
            max_length=config['EvaluationArguments']['max_token_length'],
            num_beams=model.config.num_beams,
            bos_token_id=tokenizer.pad_token_id,
            length_penalty=model.config.length_penalty,
            output_attentions=True,
            return_dict_in_generate=True
        )
        
        sequences = outputs.sequences
        
        # Replace invalid tokens with <unk>
        if len(np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
            sequences[sequences > len(tokenizer) - 1] = tokenizer.unk_token_id

        # Decode prediction
        decoded_pred = tokenizer.decode(sequences[0], skip_special_tokens=True)
        
        # Post-process the prediction
        processed_pred, _ = postprocess_text([decoded_pred], [""])
        
        return processed_pred[0]