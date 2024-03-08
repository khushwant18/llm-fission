from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from models.gpt2.custom_modeling_gpt2 import GPT2Model
from utils.load_layers import load_pretrained_embedding
from flask import Flask, request, jsonify
import requests
import json
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--layer_url_mapping', required=True, help='Layer to URL mapping')
args = parser.parse_args()

# Access the URL, layers, and layer_url_mapping arguments
layer_urls = args.layer_url_mapping
print(layer_urls)
layer_url_map = []
pairs = layer_urls.rstrip(',').split(',')

for pair in pairs:
    parts = pair.split('=')
    if len(parts) == 2:
        start, end = parts[0].split(':')
        url = parts[1]
        layer_url_map.extend([url] * (int(end) - int(start) + 1))

model_path = "Writer/palmyra-small"

wte = load_pretrained_embedding(model_path, "transformer.wte").eval().to('cpu')
wpe = load_pretrained_embedding(model_path, "transformer.wpe").eval().to('cpu')
lm_head = load_pretrained_embedding(model_path, "lm_head").eval().to('cpu')
ln_f = load_pretrained_embedding(model_path,"transformer.ln_f").eval().to('cpu')


tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
pretrained_transformer=GPT2Model(config)

def generate_text(prompt, max_len):
    input_ids = tokenizer.encode(prompt, return_tensors="pt",max_length=max_len)
    with torch.no_grad():
        for _ in range(max_len):

            transformer_outputs = pretrained_transformer(input_ids=input_ids,wte=wte,wpe=wpe,ln_f=ln_f,layer_url_map=layer_url_map)
            logits = transformer_outputs[0]
            logits = lm_head(logits)
            next_token = torch.argmax(logits[:, -1, :])
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            # Check if the generated text ends with an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return generated_text

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def func4():
    data = request.get_json()
    prompt = data['prompt']
    max_len = data['max_len']

    return generate_text(prompt, max_len)

@app.route('/wte', methods=['POST'])
def func1():
    data = request.get_json()
    inpdata = data['data']
    inpdata =  torch.tensor(inpdata)
    res = wte(inpdata)

    return jsonify({"res": res.detach().numpy().tolist()})

@app.route('/wpe', methods=['POST'])
def func2():
    data = request.get_json()
    inpdata = data['data']
    inpdata =  torch.tensor(inpdata)
    res = wpe(inpdata)

    return jsonify({"res": res.detach().numpy().tolist()})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7002)