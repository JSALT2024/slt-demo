# slt-demo
**Sign Language Translation - Gradio web app for Video to Text**

This demo is a proof of concept for the translation system of ASL. It uses YoloV8, MediaPipe and T5 in the backend, trained on the YoutubeASL dataset by the CV team at UWB. See [T5_for_SLT](https://github.com/zeleznyt/T5_for_SLT) and [PoseEstimation](https://github.com/JSALT2024/PoseEstimation).

## Find the demo at [HuggingFace spaces](https://huggingface.co/spaces/VaJavorek/slt-demo)

## How to run locally

 1. Clone repository
 ```commandline
git clone https://github.com/JSALT2024/slt-demo.git
```
    

 2. Install dependencies

```commandline
pip install -r requirements.txt
```

 3. Run the demo
 ```commandline
python app.py
```
 4. Open http://127.0.0.1:7860/

 5. You are good to go! Upload any video or use your webcam 📷


## Code structure
```
slt-demo/
├── backend.py
├── predict_pose.py
├── sltGradio.py
├── checkpoints/
│   ├── pose/
│   │   └── [pose model files]
│   └── t5-v1_1-base/
│       └── [T5 model files]
├── configs/
│   └── predict_config_demo.yaml
├── dataset/
│   └── generic_sl_dataset.py
├── model/
│   ├── configuration_t5.py
│   └── modeling_t5.py
├── utils/
│   └── translation.py
└── video/
    └── [example videos]
```