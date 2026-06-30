import gradio as gr
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from backend import process_input

# upload 1GB modelu
print("Checking for large model file...")
local_model_path = os.path.join("Uni_Sign", "unisign_model", "best_checkpoint.pth")
if os.path.exists(local_model_path):
    model_path = local_model_path
else:
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="plice13/sign-language-weights", 
        filename="best_checkpoint.pth"                  
    )
print(f"File successfully loaded at: {model_path}")
os.environ["UNISIGN_WEIGHTS"] = model_path
# ====================


def process_video(input_video_path):
    # Generate a translation in the backend.
    translation = process_input(input_video_path)
    return translation

# Custom CSS for the dark green background and centered layout
custom_css = """
body, html, gradio-app {
    background-color: darkgreen !important;
}
.gradio-container {
    background-color: darkgreen !important;
    border: none !important;
}

#center-column {
    max-width: 700px;
    margin: 0 auto;
    background-color: lightgreen;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
h1, h3 {
    text-align: center;
    color: white !important;
    font-weight: bold !important;
}
"""

with gr.Blocks(title="Sign Language Translation", css=custom_css, theme=gr.themes.Default(primary_hue="green")) as app:
    
    gr.Markdown("<h1>Sign Language to Text Translation</h1>")
    gr.Markdown("<h3>Upload an ASL video and get a text translation.</h3>")
    
    # Everything inside this column will be centered based on the CSS above
    with gr.Column(elem_id="center-column"):
        video_input = gr.Video(label="Upload a video")
        submit_btn = gr.Button("Translate", variant="primary")
        text_output = gr.Textbox(label="Translation")
            
    submit_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=text_output,
    )

if __name__ == "__main__":
    app.launch()

