from flask import Flask, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import json

from transformers import PreTrainedModel
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from huggingface_hub import HfApi



def load_owner(hf_token, owner_path):
    hf_hub_download(repo_id=owner_path, filename="pytorch_model.bin", token=hf_token,local_dir="./")
    hf_hub_download(repo_id=owner_path, filename="config.json", token=hf_token,local_dir="./")
    with open("config.json", "r") as file:
      config = json.load(file)
    # Load the model from the .bin file
    print(config)
    path = 'pytorch_model.bin'
    pretrained_layer = torch.load(path, map_location=torch.device('cpu'))
    part_user = nn.Linear(in_features=config['in_features'], out_features=config['out_features'], bias=False)
    part_user.weight.data.copy_(pretrained_layer['linear.weight'])
 
    return part_user


app = Flask(__name__)



file_path = "owner.json"

# Read the JSON file
with open(file_path, 'r') as json_file:
    loaded_data = json.load(json_file)
    
hf_token = loaded_data['HF_TOKEN']
owner_path = loaded_data['OWNER_PATH']
model_path = loaded_data['MODEL_PATH']


lm_head = load_owner(hf_token, owner_path)

@app.route('/lm_head', methods=['POST'])
def main():
    data = request.get_json()
    logits = data['logits']
    max_length = 10
    tensor_payload = torch.Tensor(logits)
    logits = lm_head(tensor_payload)

    np_array = logits.detach().numpy()

    list_res = np_array.tolist()
    
    return jsonify({"logits": list_res})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)