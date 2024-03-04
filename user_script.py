
from flask import Flask, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import json

from transformers import PreTrainedModel
from torch import nn

from transformers import AutoModel

from huggingface_hub import HfApi

import requests



def get_logits(logits, url):

    np_array = logits.detach().numpy()

    list_payload = np_array.tolist()
    # print("inside get logits", url)
    response = requests.post(url, json={"logits": list_payload})
    # print("response", response)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Try to parse the JSON content
            logits = response.json()["logits"]
            return torch.Tensor(logits)
        
        except ValueError as e:
            # Handle JSON parsing error
            print(f"Error parsing JSON: {e}")
    else:
        # Handle unsuccessful request (non-200 status code)
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)  # Print the response content for debugging purposes



def load_user(hf_token, user_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    pretrained_model = AutoModel.from_pretrained(user_path,token=hf_token)
    return tokenizer, pretrained_model

def generate_text(prompt,tokenizer,max_length,pretrained_transformer,url):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using greedy search
    output_hidden_states = None
    output_attentions=None
    with torch.no_grad():
        for _ in range(max_length):


            transformer_outputs = pretrained_transformer(input_ids)
            # break
            logits = transformer_outputs[0]
            # logits = pretrained_model.lm_head(logits)
      
            logits = get_logits(logits, url)
            # logits = logits/-0.0124

            next_token = torch.argmax(logits[:, -1, :])

            # Append the next token to the input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

            # Check if the generated text ends with an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return generated_text




app = Flask(__name__)

file_path = "./user.json"

# Read the JSON file
with open(file_path, 'r') as json_file:
    loaded_data = json.load(json_file)

hf_token = loaded_data['HF_TOKEN']
user_path = loaded_data['USER_PATH']
model_path = loaded_data['MODEL_PATH']
tokenizer, pretrained_transformer = load_user(hf_token, user_path, model_path)



@app.route('/generate', methods=['POST'])
def main():
    data = request.get_json()
    prompt = data['prompt']
    url = data['url']
    print(prompt,url)
    max_length = 10
    generated_text = generate_text(prompt,tokenizer,max_length,pretrained_transformer,url)
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)