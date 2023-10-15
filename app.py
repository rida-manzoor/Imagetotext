import torch 
import re 
import gradio as gr
from PIL import Image

from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device='cpu'

model_id = "nttdataspain/vit-gpt2-stablediffusion2-lora"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

# Predict function
def predict(image):
    img = image.convert('RGB')
    model.eval()
    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

input = gr.inputs.Image(label="Upload any Image", type = 'pil', optional=True)
output = gr.outputs.Textbox(type="text",label="Captions")
examples_folder = os.path.join(os.path.dirname(__file__), "examples")
examples = [os.path.join(examples_folder, file) for file in os.listdir(examples_folder)]

with gr.Blocks() as demo:
    
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h2 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            📸 ViT Image-to-Text with LORA 📝
        </h2>   
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 2rem; margin-bottom: 1.5rem">
        In the field of large language models, the challenge of fine-tuning has long perplexed researchers. Microsoft, however, has unveiled an innovative solution called <b>Low-Rank Adaptation (LoRA)</b>. With the emergence of behemoth models like GPT-3 boasting billions of parameters, the cost of fine-tuning them for specific tasks or domains has become exorbitant.
        <br>
        <br>
        You can find more info here: <u><a href="https://medium.com/@daniel.puenteviejo/fine-tuning-image-to-text-algorithms-with-lora-deb22aa7da27" target="_blank">Medium article</a></u>
        </h2>
        </div>
        """)
    
    with gr.Row():
            with gr.Column(scale=1):
                img = gr.inputs.Image(label="Upload any Image", type = 'pil', optional=True)
                button = gr.Button(value="Describe")
            with gr.Column(scale=1):
                out = gr.outputs.Textbox(type="text",label="Captions")   
                
    button.click(predict, inputs=[img], outputs=[out])
 
    gr.Examples(
        examples=examples,
        inputs=img,
        outputs=out,
        fn=predict,
        cache_examples=True,
    )
demo.launch(debug=True)