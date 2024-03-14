import argparse
from flask import Flask, request, jsonify
import torch
from utils.load_layers import load_pretrained_block
from utils.model_type import detect_language_model_family
from models.llama.custom_modeling_llama import LlamaModel
from transformers import AutoConfig


def parse_arguments():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--layers', nargs='+', required=True, help='Range of layer IDs')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], help='Device type to use ("cpu" or "cuda").')
    parser.add_argument('--model', help='Enter Hugging Face repo')
    return parser.parse_args()

def load_blocks(model_path, layers, device_type):
    blocks = [load_pretrained_block(model_path, b).eval().to(device_type) for b in layers]
    return blocks

def process_blocks(blocks, hidden_states,cache_position=None,position_ids=None):
    if model_type == "gpt2":
        for block in blocks:
            outputs = block(hidden_states,
                            layer_past=None,
                            attention_mask=None,
                            head_mask=None,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            use_cache=True,
                            output_attentions=False)
            hidden_states = outputs[0]
    elif model_type == "llama":
        causal_mask = llama._update_causal_mask(attention_mask=None, input_tensor=hidden_states)
        for block in blocks:
            outputs = block(hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position)
            hidden_states = outputs[0]
    return hidden_states.to('cpu').detach().numpy().tolist()

app = Flask(__name__)

@app.route('/block2', methods=['POST'])
def process_request():
    data = request.get_json()
    hidden_states = torch.tensor(data['hidden_states']).to(device_type)
    if data['cache_position'] != None and data['position_ids'] != None:
        position_ids = torch.tensor(data['position_ids']).to(device_type)
        cache_position = torch.tensor(data['cache_position']).to(device_type)
    else:
        position_ids=None
        cache_position=None
    processed_states = process_blocks(blocks, hidden_states, cache_position, position_ids)

    return jsonify({"res": processed_states})

if __name__ == '__main__':
    args = parse_arguments()

    device_type = args.device
    layers = args.layers
    model_path = args.model
    config = AutoConfig.from_pretrained(model_path)
    model_type = detect_language_model_family(config)

    start, end = map(int, layers[0].split(':'))
    layers = [str(i) for i in range(start, end + 1)]

    print("Deploying layers:", layers)
    if model_type == "llama":
        llama=LlamaModel(config) 

    blocks = load_blocks(model_path, layers, device_type)

    app.run(host='0.0.0.0', port=7001)
