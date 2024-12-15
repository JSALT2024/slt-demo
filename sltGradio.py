import cv2
import gradio as gr
from backend import dinosaurus

def process_image(input_image):
    output_sign = dinosaurus(input_image)
    return output_sign

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,  # The function to be called
    inputs=[
        gr.Image(type="numpy", sources=["upload", "webcam"], label="Upload an image or take a picture"),
    ],
    outputs=gr.Textbox(label="Predicted Sign"),
    title="Sign Language Recognition - Fingerspelling Alphanumerals in SAUDI SL",  # Title of the interface
    description="This demo is a proof of concept for the fingerspelling recognition system of the SAUDI SL. It uses MediaPipe and DINO in the backend, trained on the SAUDI SL dataset from Mohammad Alghannami and Maram Aljuaid."
)

example_images = gr.Markdown(
    """
    ## Example input
    Here are three example input images, you can drag and drop them into the image input window:
    """
)

example_image1 = gr.Image(value='gradio/A.jpg', type='numpy', label="Example Image of A", width=420, height=280)
example_image2 = gr.Image(value='gradio/1.jpg', type='numpy', label="Example Image of 1", width=420, height=280)
example_image3 = gr.Image(value='gradio/2.jpg', type='numpy', label="Example Image of 2", width=420, height=280)

# Combine the interface and example images
app = gr.Blocks()

with app:
    iface.render()
    example_images.render()
    with gr.Row():
        example_image1.render()
        example_image2.render()
        example_image3.render()

# Launch the interface
if __name__ == "__main__":
    app.launch()
