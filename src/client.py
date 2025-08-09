import argparse
import logging
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Dict, Any
from utils.transformer_components import load_transformer_components
from utils.model_type import detect_language_model_family
from models.gpt2.custom_modeling_gpt2 import GPT2Model
from models.gpt_oss.custom_modeling_gpt_oss import GptOssModel

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description='Client script for interacting with the GPT-2 model server')
    parser.add_argument('--layer_url_mapping', required=True, help='Layer to URL mapping')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device type to use ("cpu" or "cuda").')
    parser.add_argument('--model', required=True, help='Enter Hugging Face repo')
    parser.add_argument('--port', default=5000, type=int, help='Enter port number')
    return parser.parse_args()

def parse_layer_url_mapping(layer_urls: str) -> List[Dict[str, Any]]:
    """Parses a string containing layer to URL mappings into a list of dictionaries."""
    layer_url_map = []
    try:
        pairs = layer_urls.rstrip(',').split(',')
        for pair in pairs:
            parts = pair.split('=')
            if len(parts) == 2:
                layer_range, url = parts
                start, end = map(int, layer_range.split(':'))
                layer_url_map.append(url)
    except ValueError as e:
        logging.error(f"Error parsing layer URL mapping: {e}")
        raise

    return layer_url_map

def generate_text(prompt, max_len, transformer_components, pretrained_transformer, tokenizer, lm_head, layer_url_map):
    """function for generating text based on prompt using model components."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=max_len).to(device_type)

    with torch.no_grad():
        for _ in range(max_len):
            transformer_outputs = pretrained_transformer(input_ids=input_ids, transformer_components=transformer_components, layer_url_map=layer_url_map, device_type=device_type)
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
    """Endpoint for generating text based on a provided prompt and other parameters."""
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        max_len = int(data.get('max_len', 50))
        if not prompt or max_len <= 0:
            raise ValueError("Invalid input parameters.")
        
        return generate_text(prompt, max_len, transformer_components, pretrained_transformer, tokenizer, lm_head, layer_url_map)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

def configure_app():
    """Configures Flask application settings."""
    app.config['DEBUG'] = False  # Set to True for debugging, False for production

if __name__ == '__main__':
    configure_app()
    args = parse_arguments()
    try:
        layer_url_map = parse_layer_url_mapping(args.layer_url_mapping)
        device_type = args.device
        model_path = args.model
        config = AutoConfig.from_pretrained(model_path)
        model_type = detect_language_model_family(config)
        transformer_components, pretrained_transformer, tokenizer, lm_head = load_transformer_components(model_path, device_type, model_type, config)

        app.run(host='0.0.0.0', port=args.port)
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
