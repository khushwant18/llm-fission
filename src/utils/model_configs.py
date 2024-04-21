# model_configs.py

from torch import nn
from models.llama.custom_modeling_llama import LlamaRMSNorm
from models.mistral.custom_modeling_mistral import MistralRMSNorm

#configuration builder for GPT2
def get_gpt2_embedding_configs(config, embed_dim):
    return {
        "wte": {"class": nn.Embedding, "params": {"num_embeddings": config.vocab_size, "embedding_dim": embed_dim}, "prefix": "transformer.wte"},
        "wpe": {"class": nn.Embedding, "params": {"num_embeddings": config.max_position_embeddings, "embedding_dim": embed_dim}, "prefix": "transformer.wpe"},
        "ln_f": {"class": nn.LayerNorm, "params": {"normalized_shape": embed_dim, "eps": config.layer_norm_epsilon}, "prefix": "transformer.ln_f"},
        "lm_head": {"class": nn.Linear, "params": {"in_features": config.n_embd, "out_features": config.vocab_size, "bias": False}, "prefix": "transformer.wte"},
    }

#configuration builder for LLaMA
def get_llama_embedding_configs(config):
    return {
        "embed_tokens": {"class": nn.Embedding, "params": {"num_embeddings": config.vocab_size, "embedding_dim": config.hidden_size, "padding_idx": config.pad_token_id}, "prefix": "model.embed_tokens"},
        "norm": {"class": LlamaRMSNorm, "params": {"hidden_size": config.hidden_size, "eps": config.rms_norm_eps}, "prefix": "model.norm"}, 
        "lm_head": {"class": nn.Linear, "params": {"in_features": config.hidden_size, "out_features": config.vocab_size, "bias": False}, "prefix": "lm_head"},
    }

def get_mistral_embedding_configs(config):
    return {
        "embed_tokens": {"class": nn.Embedding, "params": {"num_embeddings": config.vocab_size, "embedding_dim": config.hidden_size, "padding_idx": config.pad_token_id}, "prefix": "model.embed_tokens"},
        "norm": {"class": MistralRMSNorm, "params": {"hidden_size": config.hidden_size, "eps": config.rms_norm_eps}, "prefix": "model.norm"}, 
        "lm_head": {"class": nn.Linear, "params": {"in_features": config.hidden_size, "out_features": config.vocab_size, "bias": False}, "prefix": "lm_head"},
    }
 