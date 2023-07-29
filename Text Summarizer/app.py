
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 

hf_api_key = os.environ["HF_API_KEY"]


# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL= os.environ["API_URL"]): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))


import gradio as gr
def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']
    
gr.close_all()
demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label = "Text to Summarize")],
                    outputs=[gr.Textbox(label = "Result")],
                    title="Text Summarizer with Distilbart-cnn",
                    description= "Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!")
demo.launch()