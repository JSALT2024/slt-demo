import cv2
import gradio as gr
import os
import uuid
from backend import dinosaurus

def process_video(input_video_path):
    # Generate captions in the backend function, caption per frame.
    frame_captions = dinosaurus(input_video_path)
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the input video.")
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_file = f"processed_{uuid.uuid4().hex}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Read each frame, overlay the caption, and write to output
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        caption = frame_captions[frame_index] if frame_index < len(frame_captions) else ""
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)   # white text
        thickness = 2
        
        text_size, _ = cv2.getTextSize(caption, font, font_scale, thickness)
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        cv2.putText(frame, caption, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        out.write(frame)
        
        frame_index += 1
    
    cap.release()
    out.release()

    return output_file

example_videos = [
    ["slt-demo/video/videos-1.mp4"],
    ["slt-demo/video/videos-20.mp4"],
    ["slt-demo/video/videos-25.mp4"]
]

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.Video(label="Captioned Video"),
    title="Sign Language Recognition - Video to Caption",
    description="Upload a video and get it back with captions.",
    examples=example_videos  # Provide the example videos here
)

app = gr.Blocks()

with app:
    iface.render()

if __name__ == "__main__":
    app.launch()
