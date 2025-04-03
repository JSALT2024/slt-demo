import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from transformers import T5Tokenizer
from model.configuration_t5 import SignT5Config
from transformers import T5Config
from model.modeling_t5 import T5ModelForSLT
from utils.translation import postprocess_text
import evaluate
import yaml
from dataset.generic_sl_dataset import SignFeatureDataset as DatasetForSLT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned T5 model for SLT")

    # Configuration
    parser.add_argument("--config_file", type=str, required=True, default='config.yaml')

    # Model and data paths
    parser.add_argument("--model_name", type=str, default=None, help="Model name or folder inside model_dir.")
    parser.add_argument("--output_dir", type=str, default=None)

    # Data processing
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Max number of frames for sign inputs.")
    parser.add_argument("--max_token_length", type=int, default=None, help="Max token length for labels.")
    parser.add_argument("--skip_frames", default=None)

    # Generation parameters
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the directory containing the fine-tuned model and config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference.")
    # parser.add_argument("--pose_dim", type=int, default=208, help="Dimension of the pose embeddings.")

    # Evaluation arguments
    parser.add_argument("--num_beams", type=int, default=None, help="Number of beams for beam search.")
    parser.add_argument("--length_penalty", type=float, default=None, help="Length penalty for generation.")
    parser.add_argument("--early_stopping", type=bool, default=None, help="Use early stopping in generation.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None, help="No repeat ngram size.")

    # Running arguments
    parser.add_argument("--dev", action="store_true", help="Use dev mode.")
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--is_normalized", action="store_true", help="If the data is normalized.")

    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")

    return parser.parse_args()


def load_config(cfg_path):
    """
    Load config from a yaml file. 'none' and 'None' values are replaced by None value.
    Args:
        cfg_path: Path to config file
    Returns:
        config (dict): Config
    """
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    for param, value in cfg['EvaluationArguments'].items():
        if value == 'none' or value == 'None':
            cfg['EvaluationArguments'][param] = None

    return cfg


def update_config(cfg, args):
    """
    Update config with args passed. Default None arguments are ignored.
    Args:
        cfg (dict): Config
        args (argparse.Namespace): Argument parsed from the command-line
    Returns:
        cfg (dict): Updated config
    """
    for k, v in vars(args).items():
        if k in cfg['EvaluationArguments'] and v is not None:
            cfg['EvaluationArguments'][k] = v
            if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
                print('Config value updated by args - {}: {}'.format(k, v))
    return cfg

def collate_fn(batch, max_sequence_length, max_token_length, pose_dim):
    # Check if sign_inputs is a dictionary or tensor
    if isinstance(batch[0]["sign_inputs"], dict):
        # Handle case where sign_inputs is a dictionary
        sign_inputs_list = []
        for sample in batch:
            # Extract the pose tensor from the dictionary
            if "pose" in sample["sign_inputs"]:
                pose_tensor = sample["sign_inputs"]["pose"]
                # Pad if necessary
                if pose_tensor.shape[0] < max_sequence_length:
                    padded_tensor = torch.cat((pose_tensor, 
                                              torch.zeros(max_sequence_length - pose_tensor.shape[0], pose_dim)), 
                                              dim=0)
                else:
                    padded_tensor = pose_tensor[:max_sequence_length]
                sign_inputs_list.append(padded_tensor)
            else:
                # Fallback if pose not found
                sign_inputs_list.append(torch.zeros(max_sequence_length, pose_dim))
        
        sign_inputs = torch.stack(sign_inputs_list)
    else:
        # Original code for when sign_inputs is a tensor
        sign_inputs = torch.stack([
            torch.cat((sample["sign_inputs"], 
                      torch.zeros(max_sequence_length - sample["sign_inputs"].shape[0], pose_dim)), 
                      dim=0)
            if sample["sign_inputs"].shape[0] < max_sequence_length
            else sample["sign_inputs"][:max_sequence_length]
            for sample in batch
        ])
    
    return {
        "sign_inputs": sign_inputs,
        "attention_mask": torch.stack([
            torch.cat((sample["attention_mask"], 
                      torch.zeros(max_sequence_length - sample["attention_mask"].shape[0])), 
                      dim=0)
            if sample["attention_mask"].shape[0] < max_sequence_length
            else sample["attention_mask"][:max_sequence_length]
            for sample in batch
        ]),
        "labels": torch.stack([
            torch.cat((sample["labels"].squeeze(0), 
                      torch.zeros(max_token_length - sample["labels"].shape[0])), 
                      dim=0)
            if sample["labels"].shape[0] < max_token_length
            else sample["labels"][:max_token_length]
            for sample in batch
        ]).squeeze(0).to(torch.long),
    }

def evaluate_model(model, dataloader, tokenizer, evaluation_config):
    from datetime import datetime
    def log_message(message, log_file):
        current_time = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{current_time} GMT+1] {message}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()
        
    with open("evaluation.log", "a") as log_file:
        log_message("Starting model evaluation", log_file)
        model.eval()
        predictions, labels = [], []
        
        # Determine model dtype
        model_dtype = next(model.parameters()).dtype
        
        # Get total number of batches
        total_batches = len(dataloader)
        log_message(f"Beginning inference loop with {total_batches} batches", log_file)
        
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if step % 10 == 0:
                    log_message(f"Processing batch {step}/{total_batches}", log_file)
                    
                # Convert batch to the same dtype as the model
                batch = {k: v.to(model.base_model.device).to(model_dtype) for k, v in batch.items()}
                
                if len(batch['labels'].shape) < 2:
                    batch['labels'] = batch['labels'].unsqueeze(0)
                
                outputs = model.generate(
                    **batch,
                    early_stopping=model.config.early_stopping,
                    no_repeat_ngram_size=model.config.no_repeat_ngram_size,
                    max_length=evaluation_config['max_sequence_length'],
                    num_beams=model.config.num_beams,
                    bos_token_id=tokenizer.pad_token_id,
                    length_penalty=model.config.length_penalty,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
                
                sequences = outputs.sequences
                
                # Replace invalid tokens with <unk>
                if len(np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
                    log_message(f'Replacing <unk> for illegal tokens found on indexes {np.where(sequences.cpu().numpy() > len(tokenizer) - 1)[1]}', log_file)
                sequences[sequences > len(tokenizer) - 1] = tokenizer.unk_token_id

                decoded_preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                predictions.extend(decoded_preds)
                labels.extend([[translation] for translation in decoded_labels])
        
        log_message("Completed model evaluation", log_file)
        return predictions, labels


def get_sign_input_dim(config):
    sign_input_dim = 0
    for mod in config['SignDataArguments']['visual_features']:
        if config['SignDataArguments']['visual_features'][mod]['enable_input']:
            sign_input_dim += config['SignModelArguments']['projectors'][mod]['dim']
    return sign_input_dim


def main():
    from datetime import datetime
    def log_message(message, log_file):
        current_time = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{current_time} GMT+1] {message}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    with open("evaluation.log", "a") as log_file:
        log_message("Starting script", log_file)
        args = parse_args()
        if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
            log_message('Loading config...', log_file)
        config = load_config(args.config_file)
        config = update_config(config, args)

        log_message("Loading model configuration", log_file)
        evaluation_config = config['EvaluationArguments']
        model_config = config['ModelArguments']
        model_config['sign_input_dim'] = get_sign_input_dim(config)

        # Initialize the custom model
        t5_config = SignT5Config()
        for param, value in model_config.items():
            if param not in vars(t5_config):
                log_message(f'{param} not in SignT5Config. It may be ignored...', log_file)
            t5_config.__setattr__(param, value)

        log_message("Loading model and tokenizer", log_file)
        # Load model and tokenizer
        model = T5ModelForSLT.from_pretrained(evaluation_config['model_dir'], config=t5_config)
        model.config.output_attentions = True
        for param in model.parameters(): param.data = param.data.contiguous()
        tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name, clean_up_tokenization_spaces=True)

        log_message("Preparing dataset", log_file)
        # Prepare dataset
        dataset = DatasetForSLT(tokenizer= tokenizer,
                                    sign_data_args=config['SignDataArguments'],
                                    split=evaluation_config['split'],
                                    skip_frames=evaluation_config['skip_frames'],
                                    max_token_length=evaluation_config['max_token_length'],
                                    max_sequence_length=evaluation_config['max_sequence_length'],
                                    max_samples=evaluation_config['max_val_samples'],
                                    )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=evaluation_config['batch_size'],
            collate_fn=lambda batch: collate_fn(
                batch,
                max_sequence_length=evaluation_config['max_sequence_length'],
                max_token_length=evaluation_config['max_token_length'],
                pose_dim=config['SignModelArguments']['projectors']['pose']['dim'],
            ),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Print device information
        log_message(f"Using device: {device}", log_file)
        if device == "cuda":
            log_message(f"GPU: {torch.cuda.get_device_name(0)}", log_file)
            log_message(f"CUDA available: {torch.cuda.is_available()}", log_file)
            log_message(f"CUDA device count: {torch.cuda.device_count()}", log_file)
        
        # Print model parameter information
        param = next(model.parameters())
        log_message(f"Model parameter device: {param.device}", log_file)
        log_message(f"Model parameter dtype: {param.dtype}", log_file)

        log_message("Running model evaluation", log_file)
        predictions, labels = evaluate_model(model, dataloader, tokenizer, evaluation_config)

        log_message("Post-processing predictions", log_file)
        # Postprocess predictions and references
        decoded_preds, decoded_labels = postprocess_text(predictions, [ref[0] for ref in labels])

        if args.verbose:
            for i in range(min(5, len(decoded_preds))):
                log_message(f"Prediction: {decoded_preds[i]}", log_file)
                log_message(f"Reference: {decoded_labels[i]}", log_file)
                log_message("-" * 50, log_file)

        log_message("Computing metrics", log_file)
        # Compute metrics
        sacrebleu = evaluate.load('sacrebleu')
        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "bleu": result["score"],
            'bleu-1': result['precisions'][0],
            'bleu-2': result['precisions'][1],
            'bleu-3': result['precisions'][2],
            'bleu-4': result['precisions'][3],
        }

        result = {k: round(v, 4) for k, v in result.items()}

        if args.verbose:
            for key, value in result.items():
                log_message(f"{key}: {value:.4f}", log_file)

        log_message("Saving predictions", log_file)
        # Save predictions
        all_predictions = [
            {
                "prediction": pred,
                "reference": ref
            }
            for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        all_predictions = {'metrics': result, 'predictions': all_predictions[:100]}

        os.makedirs(evaluation_config['output_dir'], exist_ok=True)
        prediction_file = os.path.join(evaluation_config['output_dir'], "predictions.json")
        with open(prediction_file, "w") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=4)

        log_message(f"Predictions saved to {prediction_file}", log_file)
        log_message("Script completed", log_file)

if __name__ == "__main__":
    main()
