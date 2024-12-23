from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from google.colab import drive
import gradio as gr
import torch
from gpt import build_chain


model_path = "blip_model_trained1"
processor = BlipProcessor.from_pretrained(model_path)
your_key="sk"
chain_model=build_chain(key=your_key)


image_to_text = pipeline(
    "image-to-text",
    model=model_path, 
    processor=processor,  
    device=0 if torch.cuda.is_available() else -1  
)


def dsc(image):
    
        result = image_to_text(image) 
        text = result[0]['generated_text']
        description = chain_model.invoke(text).content

        return description 


demo = gr.Interface(
    title="Chest X-ray Image Description",
    fn=dsc,
    inputs=gr.Image(type="pil"), 
    outputs="text"  
)

if __name__ == "__main__":
    demo.launch(share=True) 