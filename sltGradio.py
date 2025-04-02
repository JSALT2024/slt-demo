import gradio as gr
import os
from backend import dinosaurus

def process_video(input_video_path):
    # Generate a translation in the backend.
    # Assuming dinosaurus now returns a single string translation
    translation = dinosaurus(input_video_path)
    
    return translation

example_videos = [
    ["video/videos-1.mp4"],
    ["video/videos-20.mp4"],
    ["video/videos-25.mp4"]
]

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.Textbox(label="Translation"),
    title="Sign Language Recognition - Video to Text",
    description="Upload a sign language video and get the text translation.",
    examples=example_videos
)

app = gr.Blocks()

with app:
    iface.render()

if __name__ == "__main__":
    app.launch()
