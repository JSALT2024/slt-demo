import gradio as gr
import os
from backend import process_input, visualize_pose_keypoints

def process_video(input_video_path):
    # Generate a translation in the backend.
    translation = process_input(input_video_path)
    
    # Find the most recent visualization to display alongside the translation
    if os.path.exists("visualizations"):
        visualization_files = [os.path.join("visualizations", f) for f in os.listdir("visualizations") if f.endswith(".png")]
        if visualization_files:
            # Find the most recent visualization file
            latest_vis = max(visualization_files, key=os.path.getctime)
            return translation, latest_vis
    
    # Return just the translation if no visualization is found
    return translation, None

# Ensure example videos exist
example_videos = []
if os.path.exists("video"):
    potential_examples = [f for f in os.listdir("video") if f.endswith(".mp4")]
    example_videos = [["video/" + f] for f in potential_examples[:5]]  # Limit to 5 examples

# If no examples found, add a note
if not example_videos:
    example_videos = [["Please add example videos to the 'video' folder"]]

with gr.Blocks(title="Sign Language Translation") as app:
    gr.Markdown("# Sign Language to Text Translation")
    gr.Markdown("""Upload a sign language video and get a text translation. 
    The system uses MediaPipe to extract pose features and a T5 model to convert them to text.
    
    For best results:
    - Ensure the signer is clearly visible and well-lit
    - Position the camera to capture the full upper body
    - Make sure hands and face are clearly visible
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload a video")
            submit_btn = gr.Button("Translate", variant="primary")
            
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Translation")
            pose_vis = gr.Image(label="Pose Detection Visualization")
    
    gr.Examples(
        examples=example_videos,
        inputs=video_input,
        outputs=[text_output, pose_vis],
        fn=process_video,
        cache_examples=True,
    )
    
    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[text_output, pose_vis],
    )
    
    gr.Markdown("""### How it works
    1. The system extracts body, hand, and face keypoints using MediaPipe
    2. These keypoints are normalized and fed into a T5 transformer model
    3. The model generates text based on the pose sequence
    
    The visualization shows the extracted pose keypoints that were used for translation.
    """)

if __name__ == "__main__":
    app.launch()
