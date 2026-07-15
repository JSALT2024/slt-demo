import sys
print("DEBUG: App is starting...", file=sys.stderr)

import time
print("DEBUG: Importing Gradio/Streamlit...", file=sys.stderr)
import gradio as gr 

print("DEBUG: Loading transformers...", file=sys.stderr)
from transformers import AutoModel

print("DEBUG: Downloading/Loading model weights... (This might take a while)", file=sys.stderr)
# Your model loading code here...
model = AutoModel.from_pretrained("...")

print("DEBUG: Model loaded successfully!", file=sys.stderr)