from transformers import AutoConfig
from models.gpt2.custom_modeling_gpt2 import GPT2Block, GPT2Model
from models.llama.custom_modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaRMSNorm
from models.mistral.custom_modeling_mistral import MistralDecoderLayer, MistralModel
from torch import nn
from utils.model_configs import get_gpt2_embedding_configs, get_llama_embedding_configs, get_mistral_embedding_configs 

def detect_language_model_family(config):

    # Extract the model type from the configuration
    model_type = config.model_type.lower()
    return model_type

def load_model_block(config, model_type,block_index=None):
    if model_type == "gpt2":
        block = GPT2Block(config)
    elif model_type == "llama":
        block = LlamaDecoderLayer(config,int(block_index))
    elif model_type == "mistral":
        block = MistralDecoderLayer(config,int(block_index))
    return block

def load_full_model(config, model_type):
    if model_type == "gpt2":
        block = GPT2Model(config)
    elif model_type == "llama":
        block = LlamaModel(config)
    elif model_type == "mistral":
        block = MistralModel(config)
    return block

def get_block_prefix(block_index, model_type):
    if model_type == "gpt2":
        block_prefix = f"transformer.h.{block_index}"
    elif model_type == "llama" or model_type == "mistral":
        block_prefix = f"model.layers.{block_index}"
    return block_prefix



def get_embedding_layer(config, embed_dim, emb_type, model_type):
    # Determine which configuration builder to use
    if model_type == "gpt2":
        embedding_configs = get_gpt2_embedding_configs(config, embed_dim)
    elif model_type == "llama":
        embedding_configs = get_llama_embedding_configs(config)
    elif model_type == "mistral":
        embedding_configs = get_mistral_embedding_configs(config)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'.")

    # Retrieve the specific configuration
    model_emb_config = embedding_configs.get(emb_type, None)

    if model_emb_config:
        embed_layer = model_emb_config["class"](**model_emb_config["params"])
        embedding_prefix = model_emb_config["prefix"]
        return embed_layer, embedding_prefix
    else:
        raise ValueError(f"Unsupported embedding type '{emb_type}' for model type '{model_type}'.")


