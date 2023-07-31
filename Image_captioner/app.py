import os
import io
import IPython.display
from PIL import Image
from transformers import pipeline
import gradio as gr

hf_api_key = os.environ["HF_API_KEY"]

get_completion = pipeline(os.environ["TASK"],model= os.environ["MODEL"])

def captioner(image):
    result = get_completion(image)
    return result[0]['generated_text']

gr.close_all()
demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                    examples=["bridge.jpg", "Sea.jpg", "Umbrella.jpg"])
                   

demo.launch()
