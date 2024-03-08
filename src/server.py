from utils.load_layers import load_pretrained_block
from flask import Flask, request, jsonify
import torch
import argparse

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--layers', type=int, nargs='+', required=True, help='Range of layer IDs')
args = parser.parse_args()

# Access the URL and layers arguments

layers = args.layers
print("deploying layers: ",layers)
model_path = "Writer/palmyra-small"

blocks = []
for b in layers:
    blocks.append(load_pretrained_block(model_path, b))

app = Flask(__name__)

@app.route('/block', methods=['POST'])
def func():
    data = request.get_json()
    states = data['hidden_states']
    i = data['index']
    hidden_states = torch.tensor(states)

    outputs=blocks[i](hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=True,
            output_attentions=False,)
    hidden_states = outputs[0]
    
    res0 = hidden_states.detach().numpy().tolist()

    return jsonify({"res": res0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7001)
