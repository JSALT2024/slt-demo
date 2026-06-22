import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import tvého nového modelu a pomocných funkcí
from Uni_Sign.models import Uni_Sign
from Uni_Sign.datasets import load_part_kp_YTASL, YTASL_GROUP_SIZES, _fill_missing_landmarks, select_frame_indices
from predict_pose import create_mediapipe_models, predict_pose, load_video_cv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
load_dotenv()

# Globální proměnné pro modely, ať se nenačítají při každém videu znovu
model = None
pose_models = None
args = None

class InferenceConfig:
    """Čistá konfigurace pro Uni_Sign inferenci"""
    def __init__(self):
        # Nezapomeň upravit cestu ke svým natrénovaným váhám
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
    """Inicializuje Uni_Sign a Mediapipe/YOLO modely pro extrakci pose."""
    global model, pose_models, args
    
    if model is not None and pose_models is not None:
        return
    
    print("Inicializuji Uni_Sign model...")
    args = InferenceConfig()
    model = Uni_Sign(args=args)
    
    # Načtení vah
    if args.finetune and os.path.exists(args.finetune):
        print(f"Načítám váhy z: {args.finetune}")
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"VAROVÁNÍ: Checkpoint '{args.finetune}' nebyl nalezen!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("Inicializuji Mediapipe a YOLO modely...")
    pose_checkpoint_folder = 'checkpoints/pose/'
    pose_models = create_mediapipe_models(pose_checkpoint_folder)

def process_pose_data_in_memory(pose_results, args):
    """
    Nahrazuje starý 'process_single_json'.
    Zpracovává 'cropped_keypoints' přímo ze slovníku (v paměti).
    """
    raw_pose = pose_results.get('cropped_keypoints', [])
    if not raw_pose:
        raise ValueError("Video neobsahuje žádné detekované pose (cropped_keypoints).")

    # NOVÝ KÓD: Převod numpy polí na čisté [X, Y] seznamy a ošetření chybějících částí těla
    pose = []
    for frame_data in raw_pose:
        formatted_frame = {}
        for part, expected_size in YTASL_GROUP_SIZES.items():
            kps = frame_data.get(part, [])
            
            # Pokud část chybí (prázdný list/array), pošleme prázdno, ať si s tím poradí _fill_missing_landmarks
            # Pokud část chybí (prázdný list/array), pošleme prázdno
            # Pokud část chybí (prázdný list/array), pošleme prázdno
            # Bezpečná kontrola Numpy pole:
            if kps is None or len(kps) == 0:
                 formatted_frame[part] = [] 
            else:
                 # Neprůstřelné řešení: převedeme data (ať už jsou cokoliv) na Numpy pole, 
                 # ořízneme první dva sloupce a převedeme zpět na čistý list.
                 formatted_frame[part] = np.array(kps)[:, :2].tolist()
        pose.append(formatted_frame)

    # ... Zbytek funkce pokračuje beze změny:

    duration = len(pose)
    tmp = select_frame_indices(duration, args.max_length, phase='test')
    skeletons = [pose[i] for i in tmp]

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

    kps_with_scores = load_part_kp_YTASL(skeletons, confs, args.normalization, args.layout)

    src_input = {}
    for key, val in kps_with_scores.items():
        src_input[key] = val.unsqueeze(0)

    seq_len = src_input['body'].shape[1]
    mask_gen = torch.ones([seq_len]) + 7
    src_input['attention_mask'] = (mask_gen != 0).long().unsqueeze(0)
    src_input['src_length_batch'] = torch.LongTensor([seq_len])
    src_input['name_batch'] = ["gradio_video"]

    return src_input


def process_input(input_video_path):
    """
    Hlavní vstupní bod pro Gradio. Spustí extrakci dat, vizualizaci 
    a inference přes Uni_Sign.
    """
    try:
        start_time = time.time()
        
        print(f"0. Začínáme inicializací... [0.00 s od startu]")
        initialize_model()
        
        print(f"1. Extrahuji keypoints z videa... [{time.time() - start_time:.2f} s od startu]")
        video_frames, _ = load_video_cv(input_video_path)
        pose_results = predict_pose(video_frames, pose_models)
        
        print(f"2. Prostor pro vizualizaci... [{time.time() - start_time:.2f} s od startu]")

        print(f"3. Přenáším data do Tenzorů pro Uni_Sign... [{time.time() - start_time:.2f} s od startu]")
        src_input = process_pose_data_in_memory(pose_results, args)
        
        # Přesunutí Tenzorů na stejné zařízení (GPU/CPU) jako model
        device = next(model.parameters()).device
        for key in ['body', 'left', 'right', 'face_all', 'attention_mask']:
            if key in src_input:
                # Ošetření typu dat na float
                if key != 'attention_mask':
                    src_input[key] = src_input[key].to(device, dtype=torch.float32)
                else:
                    src_input[key] = src_input[key].to(device)
                    
        tgt_input = {'gt_sentence': [""], 'gt_gloss': [""]}
        
        
        print(f"4. Generuji překlad... [{time.time() - start_time:.2f} s od startu]")
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
            result = "Model nedokázal generovat text. Zkus jiné video."
        
        print(f"5. VŠE HOTOVO! [{time.time() - start_time:.2f} s od startu]")
        return result
                
    except Exception as e:
        print(f"Chyba při zpracování: {e}")
        import traceback
        traceback.print_exc()
        return f"Chyba při zpracování videa: {str(e)}"
