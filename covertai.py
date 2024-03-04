from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import jsonify
import json

from transformers import PreTrainedModel
from torch import nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from huggingface_hub import HfApi
import os
from dotenv import load_dotenv




class MyPretrainedLayerWrapper(nn.Module,PyTorchModelHubMixin):
    def __init__(self, pretrained_layer):
        super().__init__()
        self.linear = pretrained_layer

    def forward(self, x):
        return self.linear(x)

def split_model(model_path, owner_path, user_path, hf_token):


    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Define the original layer
    original_layer = pretrained_model.lm_head

    # Part 1: Save the transformer part
    part_owner = pretrained_model.transformer
    part_owner.push_to_hub(user_path,token=hf_token)
    tokenizer.push_to_hub(user_path,token=hf_token)

    # Part 2: Create and save the user part
    part_user = nn.Linear(in_features=original_layer.weight.shape[1], out_features=original_layer.weight.shape[0], bias=False)
    part_user.weight.data.copy_(original_layer.weight.data)

    my_pretrained_wrapper = MyPretrainedLayerWrapper(part_user)

    my_pretrained_wrapper.push_to_hub(owner_path,token=hf_token)
    in_features = original_layer.weight.shape[1]
    out_features = original_layer.weight.shape[0]
    config={"in_features":in_features,"out_features":out_features}

    json_file_path = "config.json"
    with open(json_file_path, "w") as json_file:
        json.dump(config, json_file)
    print(json_file_path)

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj="./config.json",
        path_in_repo="config.json",
        repo_id=owner_path,
        repo_type="model",
    )
    return "success"

def user_command(user_path, model_path, hf_token):

    info = {
            "MODEL_PATH": model_path,
            "USER_PATH": user_path,
            "HF_TOKEN": ""
        }
    file_path = "user.json"

    # Write the JSON object to the file
    with open(file_path, 'w') as json_file:
        json.dump(info, json_file, indent=2) 


    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj="./user.json",
        path_in_repo="user.json",
        repo_id=user_path,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="./user_script.py",
        path_in_repo="user_script.py",
        repo_id=user_path,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="./docker_user",
        path_in_repo="docker_user",
        repo_id=user_path,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="./requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=user_path,
        repo_type="model",
    )
    enfile = f'hf_hub_download(repo_id="{user_path}", filename="user.json", token=hf_token,local_dir="./")'
    script = f'hf_hub_download(repo_id="{user_path}", filename="user_script.py", token=hf_token,local_dir="./")'
    docker = f'hf_hub_download(repo_id="{user_path}", filename="docker_user", token=hf_token,local_dir="./")'
    req = f"hf_hub_download(repo_id='{user_path}', filename='requirements.txt', token=hf_token,local_dir='./')"
    network_start = "docker network create mynetwork" 
    docker_build = "docker build -t docker_user -f ./docker_user . "
    docker_run = "docker run --name mycontainer1 --network mynetwork -p 6000:6000 -it docker_user"
    command1 = "from huggingface_hub import hf_hub_download \n" + enfile +" \n "+script+" \n "+docker+ " \n "+ req
    command2 = network_start +" && "+docker_build +" && "+docker_run
    return {"command0":command1,"command1": "jq --arg new_token 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 '.HF_TOKEN = $new_token' user.json > tmp_user.json && mv tmp_user.json user.json", "command2": command2}


def owner_command(owner_path, model_path, hf_token):

    info = {
            "MODEL_PATH": model_path,
            "OWNER_PATH": owner_path,
            "HF_TOKEN": ""
        }
    file_path = "owner.json"

    # Write the JSON object to the file
    with open(file_path, 'w') as json_file:
        json.dump(info, json_file, indent=2) 


    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj="./owner.json",
        path_in_repo="owner.json",
        repo_id=owner_path,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="./owner_script.py",
        path_in_repo="owner_script.py",
        repo_id=owner_path,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="./docker_owner",
        path_in_repo="docker_owner",
        repo_id=owner_path,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj="./requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=owner_path,
        repo_type="model",
    )
    enfile = f"hf_hub_download(repo_id='{owner_path}', filename='owner.json', token=hf_token,local_dir='./')"
    script = f"hf_hub_download(repo_id='{owner_path}', filename='owner_script.py', token=hf_token,local_dir='./')"
    docker = f"hf_hub_download(repo_id='{owner_path}', filename='docker_owner', token=hf_token,local_dir='./')"
    req = f"hf_hub_download(repo_id='{owner_path}', filename='requirements.txt', token=hf_token,local_dir='./')"
    network_start = "docker network create mynetwork" 
    docker_build = "docker build -t docker_owner -f ./docker_owner . "
    docker_run = "docker run --name mycontainer2 --network mynetwork -p 8000:8000 -it docker_owner"
    command1 = "from huggingface_hub import hf_hub_download \n" + enfile +" \n "+script+" \n "+docker+ " \n "+ req
    command2 = network_start +" && "+docker_build +" && "+docker_run
    output = {"command0": command1,"command1": "jq --arg new_token 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 '.HF_TOKEN = $new_token' owner.json > tmp_owner.json && mv tmp_owner.json owner.json", "command2": command2}
    return output
