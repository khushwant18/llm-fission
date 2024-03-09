import argparse
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from models.gpt2.custom_modeling_gpt2 import GPT2Model
from utils.load_layers import load_pretrained_embedding

app = Flask(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Client script for interacting with the GPT-2 model server')
    parser.add_argument('--layer_url_mapping', required=True, help='Layer to URL mapping')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device type to use ("cpu" or "cuda").')
    parser.add_argument('--model', help='Enter Hugging Face repo')
    return parser.parse_args()

def parse_layer_url_mapping(layer_urls):
    layer_url_map = []
    pairs = layer_urls.rstrip(',').split(',')

    for pair in pairs:
        parts = pair.split('=')
        if len(parts) == 2:
            start, end = map(int, parts[0].split(':'))
            url = parts[1]
            layer_url_map.append(url)

    return layer_url_map

def load_transformer_components(model_path, device_type):
    wte = load_pretrained_embedding(model_path, "transformer.wte").eval().to(device_type)
    wpe = load_pretrained_embedding(model_path, "transformer.wpe").eval().to(device_type)
    lm_head = load_pretrained_embedding(model_path, "lm_head").eval().to(device_type)
    ln_f = load_pretrained_embedding(model_path, "transformer.ln_f").eval().to(device_type)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    pretrained_transformer = GPT2Model(config)

    return wte, wpe, lm_head, ln_f, tokenizer, pretrained_transformer

def generate_text(prompt, max_len, transformer_components, layer_url_map):
    wte, wpe, lm_head, ln_f, tokenizer, pretrained_transformer = transformer_components
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=max_len).to(device_type)

    with torch.no_grad():
        for _ in range(max_len):
            transformer_outputs = pretrained_transformer(input_ids=input_ids, wte=wte, wpe=wpe, ln_f=ln_f, layer_url_map=layer_url_map)
            logits = transformer_outputs[0]
            logits = lm_head(logits)
            next_token = torch.argmax(logits[:, -1, :])
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return generated_text

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.get_json()
    prompt = data['prompt']
    max_len = data['max_len']

    return generate_text(prompt, max_len, transformer_components, layer_url_map)


if __name__ == '__main__':
    args = parse_arguments()
    layer_url_map = parse_layer_url_mapping(args.layer_url_mapping)
    device_type = args.device
    model_path = args.model

    transformer_components = load_transformer_components(model_path, device_type)

    app.run(host='0.0.0.0', port=7002)
