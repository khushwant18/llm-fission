import argparse
from flask import Flask, request, jsonify
import torch
from utils.load_layers import load_pretrained_block

def parse_arguments():
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--layers', type=int, nargs='+', required=True, help='Range of layer IDs')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], help='Device type to use ("cpu" or "cuda").')
    parser.add_argument('--model', help='Enter Hugging Face repo')
    return parser.parse_args()

def load_blocks(model_path, layers, device_type):
    blocks = [load_pretrained_block(model_path, b).eval().to(device_type) for b in layers]
    return blocks

def process_blocks(blocks, hidden_states):
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
    return hidden_states.detach().numpy().tolist()

app = Flask(__name__)

@app.route('/block2', methods=['POST'])
def process_request():
    data = request.get_json()
    hidden_states = torch.tensor(data['hidden_states'])

    processed_states = process_blocks(blocks, hidden_states)

    return jsonify({"res": processed_states})

if __name__ == '__main__':
    args = parse_arguments()

    device_type = args.device
    layers = args.layers
    model_path = args.model

    print("Deploying layers:", layers)

    blocks = load_blocks(model_path, layers, device_type)

    app.run(host='0.0.0.0', port=7001)
