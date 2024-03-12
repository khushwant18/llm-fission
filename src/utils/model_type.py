from transformers import AutoConfig
from models.gpt2.custom_modeling_gpt2 import GPT2Block, GPT2Model
from models.llama.custom_modeling_llama import LlamaDecoderLayer, LlamaModel
from torch import nn

def detect_language_model_family(config):

    # Extract the model type from the configuration
    model_type = config.model_type.lower()
    return model_type

def load_model(config, model_type):
    if model_type == "gpt2":
        block = GPT2Block(config)
    elif model_type == "llama":
        block = LlamaDecoderLayer(config)
    return block

def load_full_model(config, model_type):
    if model_type == "gpt2":
        block = GPT2Model(config)
    elif model_type == "llama":
        block = LlamaModel(config)
    return block

def get_block_prefix(block_index, model_type):
    if model_type == "gpt2":
        block_prefix = f"transformer.h.{block_index}"
    elif model_type == "llama":
        block_prefix = f"model.layers.{block_index}"
    return block_prefix

def get_embedding_layer(config,embed_dim, emb_type, model_type):
    if model_type == "gpt2":
        if emb_type == 'wte':
            embed_layer = nn.Embedding(config.vocab_size, embed_dim)
            embedding_prefix = "transformer.wte"
        elif emb_type == 'wpe':
            embed_layer = nn.Embedding(config.max_position_embeddings, embed_dim)
            embedding_prefix = "transformer.wpe"
        elif emb_type == 'ln_f':
            embed_layer = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
            embedding_prefix = "transformer.ln_f"
        elif emb_type == 'lm_head':
            embedding_prefix = "transformer.wte"
            embed_layer = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    if model_type =="llama":
        if emb_type == "embed_tokens":
            embed_layer = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
            embedding_prefix = "model.embed_tokens"
        elif emb_type == 'norm':
            embed_layer = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)
            embedding_prefix = "model.norm"
        elif emb_type == 'lm_head':
            embedding_prefix = "model.embed_tokens"
            embed_layer = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    

    return embed_layer, embedding_prefix