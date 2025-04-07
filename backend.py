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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
import re

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

def is_valid_sentence(text):
    """
    Check if the decoded text appears to be a valid sentence.
    
    Args:
        text: The decoded text to check
        
    Returns:
        bool: True if it looks like a valid sentence
    """
    # Remove any odd repetitions that might indicate an issue
    text = text.strip().lower()
    
    # Check for excessive repetition
    words = text.split()
    if len(words) < 2:
        return False
        
    # If more than 75% of words are the same, it's probably not valid
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
        
    most_common_word_count = max(word_counts.values())
    if most_common_word_count / len(words) > 0.75:
        print(f"Text rejected due to excessive repetition: {text}")
        return False
        
    # Check for odd character patterns (like 'ee' repeating)
    if re.search(r'([a-z])\1{3,}', text):  # Same character repeated 4+ times
        print(f"Text rejected due to character repetition: {text}")
        return False
        
    # Check if it contains common English words to ensure it's somewhat sensible
    common_english_words = {'the', 'a', 'an', 'to', 'and', 'is', 'in', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'with', 'as', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'use', 'word', 'how', 'each', 'which', 'she', 'do', 'time', 'if', 'will', 'way', 'about', 'many', 'then', 'them', 'would', 'write', 'like', 'so', 'these', 'her', 'long', 'make', 'thing', 'see', 'him', 'two', 'has', 'look', 'more', 'day', 'could', 'go', 'come', 'did', 'number', 'sound', 'no', 'most', 'people', 'my', 'over', 'know', 'water', 'than', 'call', 'first', 'who', 'may', 'down', 'side', 'been', 'now', 'find'}
    if not any(word in common_english_words for word in words):
        print(f"Text rejected due to lack of common words: {text}")
        return False
    
    # If it passes all checks, consider it valid
    return True

def normalize_keypoints(keypoints, image_width, image_height):
    """
    Normalize keypoints to [-1, 1] range.
    
    Args:
        keypoints: Numpy array of keypoints with shape (..., 2) for x,y coordinates
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Normalized keypoints
    """
    if len(keypoints) == 0:
        return keypoints
        
    normalized = keypoints.copy()
    # X coordinates: normalize to [-1, 1]
    normalized[:, 0] = (normalized[:, 0] / image_width) * 2 - 1
    # Y coordinates: normalize to [-1, 1]
    normalized[:, 1] = (normalized[:, 1] / image_height) * 2 - 1
    
    # Also normalize Z values to a similar range if present
    if normalized.shape[1] > 2:
        # Z values are typically already normalized in MediaPipe, but let's ensure they're in a reasonable range
        # Clip to a range similar to x and y for consistency
        normalized[:, 2] = np.clip(normalized[:, 2], -1.0, 1.0)
    
    # Visibility values should be in [0, 1]
    if normalized.shape[1] > 3:
        normalized[:, 3] = np.clip(normalized[:, 3], 0.0, 1.0)
    
    return normalized

def flatten_keypoints(keypoints_dict, image_width, image_height):
    """
    Flatten keypoints from all body parts into a single feature vector
    and normalize them.
    
    Args:
        keypoints_dict: Dict with pose, hand, and face landmarks
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Flattened array of keypoints
    """
    flattened = []
    
    # Process pose keypoints (33 keypoints × 4 values)
    if len(keypoints_dict['pose_landmarks']) > 0:
        pose_kp = np.array(keypoints_dict['pose_landmarks'])
        # Normalize the x,y coordinates
        pose_kp[:, :2] = normalize_keypoints(pose_kp[:, :2], image_width, image_height)
        flattened.extend(pose_kp.flatten())
    else:
        flattened.extend(np.zeros(33 * 4))
    
    # Process right hand keypoints (21 keypoints × 4 values)
    if len(keypoints_dict['right_hand_landmarks']) > 0:
        right_hand_kp = np.array(keypoints_dict['right_hand_landmarks'])
        # Normalize the x,y coordinates
        right_hand_kp[:, :2] = normalize_keypoints(right_hand_kp[:, :2], image_width, image_height)
        flattened.extend(right_hand_kp.flatten())
    else:
        flattened.extend(np.zeros(21 * 4))
    
    # Process left hand keypoints (21 keypoints × 4 values)
    if len(keypoints_dict['left_hand_landmarks']) > 0:
        left_hand_kp = np.array(keypoints_dict['left_hand_landmarks'])
        # Normalize the x,y coordinates
        left_hand_kp[:, :2] = normalize_keypoints(left_hand_kp[:, :2], image_width, image_height)
        flattened.extend(left_hand_kp.flatten())
    else:
        flattened.extend(np.zeros(21 * 4))
    
    return np.array(flattened)

def visualize_pose_keypoints(video_path, save_path=None):
    """
    Visualize pose keypoints from the input video for debugging purposes.
    
    Args:
        video_path: Path to the input video
        save_path: Path to save visualization (if None, will create based on video name)
        
    Returns:
        Path to saved visualization
    """
    global pose_models
    
    # Ensure pose models are initialized
    if pose_models is None:
        initialize_model()
    
    # Load video frames
    video_frames, _ = load_video_cv(video_path)
    
    # Get pose predictions
    pose_results = predict_pose(video_frames, pose_models)
    
    # Create visualization directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    if save_path is None:
        # Generate unique filename based on video name and timestamp
        video_name = os.path.basename(video_path).split('.')[0]
        timestamp = int(time.time())
        save_path = f"visualizations/{video_name}_{timestamp}.png"
    
    # Choose frames to visualize (first, middle, last)
    num_frames = len(pose_results['cropped_keypoints'])
    if num_frames == 0:
        return "No frames detected"
    
    frame_indices = [0]
    if num_frames > 1:
        frame_indices.append(num_frames // 2)
    if num_frames > 2:
        frame_indices.append(num_frames - 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(frame_indices), 2, figsize=(16, 6 * len(frame_indices)))
    if len(frame_indices) == 1:
        axes = axes.reshape(1, -1)
    
    # Store diagnostic statistics for feature data
    total_keypoints = {'pose': 0, 'left_hand': 0, 'right_hand': 0}
    non_zero_features = 0
    
    for i, frame_idx in enumerate(frame_indices):
        # Original image
        axes[i, 0].imshow(pose_results['images'][frame_idx])
        axes[i, 0].set_title(f"Original Frame {frame_idx}")
        axes[i, 0].axis('off')
        
        # Cropped image with keypoints
        cropped_img = pose_results['cropped_images'][frame_idx]
        keypoints = pose_results['cropped_keypoints'][frame_idx]
        
        axes[i, 1].imshow(cropped_img)
        axes[i, 1].set_title(f"Cropped Frame {frame_idx} with Keypoints")
        
        # Plot keypoints with explicit colors
        if len(keypoints['pose_landmarks']) > 0:
            kp_array = np.array(keypoints['pose_landmarks'])
            total_keypoints['pose'] += 1
            for j in range(len(kp_array)):
                if kp_array[j, 3] > 0.2:  # Only draw visible keypoints
                    circle = plt.Circle((kp_array[j, 0], kp_array[j, 1]), 3, color='blue', alpha=kp_array[j, 3])
                    axes[i, 1].add_patch(circle)
                    non_zero_features += 1
                    
        if len(keypoints['left_hand_landmarks']) > 0:
            kp_array = np.array(keypoints['left_hand_landmarks'])
            total_keypoints['left_hand'] += 1
            for j in range(len(kp_array)):
                if kp_array[j, 3] > 0.2:  # Only draw visible keypoints
                    circle = plt.Circle((kp_array[j, 0], kp_array[j, 1]), 3, color='green', alpha=kp_array[j, 3])
                    axes[i, 1].add_patch(circle)
                    non_zero_features += 1
                    
        if len(keypoints['right_hand_landmarks']) > 0:
            kp_array = np.array(keypoints['right_hand_landmarks'])
            total_keypoints['right_hand'] += 1
            for j in range(len(kp_array)):
                if kp_array[j, 3] > 0.2:  # Only draw visible keypoints
                    circle = plt.Circle((kp_array[j, 0], kp_array[j, 1]), 3, color='red', alpha=kp_array[j, 3])
                    axes[i, 1].add_patch(circle)
                    non_zero_features += 1
        
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Print diagnostic information
    print(f"Keypoint statistics in visualization:")
    print(f"  - Pose keypoints: {total_keypoints['pose']} frames with data")
    print(f"  - Left hand keypoints: {total_keypoints['left_hand']} frames with data")
    print(f"  - Right hand keypoints: {total_keypoints['right_hand']} frames with data")
    print(f"  - Total non-zero features: {non_zero_features}")
    
    return save_path

def compare_features_stats(features):
    """
    Print diagnostic statistics about the extracted features
    to help debug model issues.
    
    Args:
        features: The extracted features tensor
    """
    # Check for NaNs or infinities
    nan_count = torch.isnan(features).sum().item()
    inf_count = torch.isinf(features).sum().item()
    
    # Get basic statistics
    mean_val = features.mean().item()
    std_val = features.std().item()
    min_val = features.min().item()
    max_val = features.max().item()
    
    # Check range distribution
    in_range_minus1_1 = ((features >= -1.0) & (features <= 1.0)).float().mean().item() * 100
    zeros = (features == 0.0).float().mean().item() * 100
    
    print("\nFeature Statistics:")
    print(f"  - Shape: {features.shape}")
    print(f"  - NaN values: {nan_count}")
    print(f"  - Inf values: {inf_count}")
    print(f"  - Mean: {mean_val:.6f}")
    print(f"  - Std Dev: {std_val:.6f}")
    print(f"  - Min: {min_val:.6f}")
    print(f"  - Max: {max_val:.6f}")
    print(f"  - % in [-1, 1] range: {in_range_minus1_1:.2f}%")
    print(f"  - % zeros: {zeros:.2f}%")

def preprocess_and_standardize_features(features):
    """
    Apply additional preprocessing to features to match what the model expects.
    
    Args:
        features: torch.Tensor of shape [seq_len, feat_dim]
    
    Returns:
        Preprocessed features
    """
    # Feature statistics from evaluation data might differ from our extracted features
    # Let's standardize to have similar statistics to what the model expects
    
    # 1. Replace any NaN or Inf values
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 2. Clip values to a reasonable range
    features = torch.clamp(features, -3.0, 3.0)
    
    # 3. Ensure all values for visibility are positive (in case normalization made them negative)
    # Assuming visibility features are at every 4th position
    for i in range(3, features.shape[1], 4):
        features[:, i] = torch.abs(features[:, i])
    
    return features

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
    
    valid_frames = 0
    for i in range(num_frames):
        keypoints = pose_results['cropped_keypoints'][i]
        
        # Check if we have valid keypoints (at least some pose landmarks)
        if len(keypoints['pose_landmarks']) == 0:
            continue
            
        # Get image dimensions for normalization
        cropped_image = pose_results['cropped_images'][i]
        image_height, image_width = cropped_image.shape[:2]
        
        # Flatten and normalize keypoints
        flat_keypoints = flatten_keypoints(keypoints, image_width, image_height)
        
        # Ensure the features have the correct dimensions
        if len(flat_keypoints) > pose_dim:
            flat_keypoints = flat_keypoints[:pose_dim]
        elif len(flat_keypoints) < pose_dim:
            flat_keypoints = np.pad(flat_keypoints, (0, pose_dim - len(flat_keypoints)))
            
        features[i] = torch.tensor(flat_keypoints, dtype=torch.float32)
        attention_mask[i] = 1.0  # Mark this frame as valid
        valid_frames += 1
    
    print(f"Processed {valid_frames} valid frames out of {num_frames} total frames")
    
    features = preprocess_and_standardize_features(features)
    
    # Print diagnostic information about the features
    compare_features_stats(features)
    
    sign_inputs = {
        'pose': features,
        'mae': None,
        'dino': None,
        'sign2vec': None
    }
    
    return sign_inputs, attention_mask

def process_input(input_video_path):
    """
    Process a single video input and return the translated text.
    
    Args:
        input_video_path: Path to the input video file
        
    Returns:
        str: The translated text
    """
    try:
        # Initialize model if not already done
        initialize_model()
        
        # Visualize pose keypoints for debugging
        vis_path = visualize_pose_keypoints(input_video_path)
        print(f"Pose keypoints visualization saved to: {vis_path}")
        
        # Extract pose features from the video
        sign_inputs, attention_mask = extract_pose_features(input_video_path)
        
        # Check if we have valid frames
        valid_frame_count = attention_mask.sum().item()
        if valid_frame_count == 0:
            return "No valid pose detected in the video. Please try another video."
        
        print(f"Found {valid_frame_count} valid frames with pose data")
        
        # Prepare input batch
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        # The model expects just the pose features, which is what we'll use
        batch = {
            "sign_inputs": sign_inputs['pose'].unsqueeze(0).to(device).to(model_dtype),
            "attention_mask": attention_mask.unsqueeze(0).to(device).to(model_dtype),
            # No labels needed for inference
        }
        
        # Get generation parameters directly from the model config
        generation_params = {
            "max_length": config['EvaluationArguments']['max_token_length'],
            "num_beams": model.config.num_beams,
            "early_stopping": model.config.early_stopping,
            "length_penalty": model.config.length_penalty,
            "do_sample": model.config.do_sample,
            "temperature": model.config.temperature if hasattr(model.config, "temperature") else 1.0,
            "top_k": model.config.top_k if hasattr(model.config, "top_k") else None,
            "top_p": model.config.top_p if hasattr(model.config, "top_p") else None,
            "bos_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True
        }
        
        # Remove None values
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        print(f"Using generation parameters: {generation_params}")
        
        # Generate translation
        print("Generating translation...")
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                **generation_params
            )
            
            sequences = outputs.sequences
            print(f"Generated sequences shape: {sequences.shape}")
            print(f"First sequence tokens: {sequences[0]}")
            
            # Check for very short sequences (likely just special tokens)
            if sequences.shape[1] <= 4:
                print("Warning: Very short sequence detected, likely problematic")
            
            # Replace invalid tokens with <unk>
            if len(np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
                sequences[sequences > len(tokenizer) - 1] = tokenizer.unk_token_id

            # Decode prediction
            decoded_pred = tokenizer.decode(sequences[0], skip_special_tokens=True)
            print(f"Raw decoded prediction: '{decoded_pred}'")
            
            # If we got a very short output, let the user see it anyway
            if len(decoded_pred.strip()) < 10 and sequences.shape[1] > 4:
                print(f"Output is very short, but seems valid: '{decoded_pred}'")
                return decoded_pred
            
            # If output is empty or too short, try a more direct approach
            if not decoded_pred.strip() or sequences.shape[1] <= 4:
                print("First generation attempt produced minimal output, trying with different parameters...")
                
                # Try with more conservative parameters
                fallback_params = {
                    "max_length": config['EvaluationArguments']['max_token_length'],
                    "num_beams": 5,  # Use consistent beam size
                    "do_sample": False,  # No sampling
                    "temperature": 1.0,
                    "length_penalty": 0.6,
                    "early_stopping": True,
                    "bos_token_id": tokenizer.pad_token_id,
                    "return_dict_in_generate": True
                }
                
                outputs = model.generate(
                    **batch,
                    **fallback_params
                )
                
                sequences = outputs.sequences
                print(f"Fallback generation sequences shape: {sequences.shape}")
                print(f"Fallback sequence tokens: {sequences[0]}")
                
                if len(np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
                    sequences[sequences > len(tokenizer) - 1] = tokenizer.unk_token_id
                    
                decoded_pred = tokenizer.decode(sequences[0], skip_special_tokens=True)
                print(f"Fallback decoded prediction: '{decoded_pred}'")
                
                # If still no good output, tell the user
                if not decoded_pred.strip() or sequences.shape[1] <= 4:
                    return "The model detected sign language but couldn't produce a reliable translation."
            
        # Post-process the prediction
        processed_pred, _ = postprocess_text([decoded_pred], [""])
        
        # Return the result, even if it's short
        result = processed_pred[0].strip()
        if not result:
            return "Could not translate the sign language. Please try another video or ensure the signer is clearly visible."
        
        return result
            
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing video: {str(e)}"