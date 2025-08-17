from transformers import AutoTokenizer, AutoModel
from utils.load_layers import load_pretrained_embedding
from utils.model_type import load_full_model

class TransformerComponents:
    def __init__(self, model_path, device_type,model_type,config):
        self.model_path = model_path
        self.device_type = device_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pretrained_transformer = load_full_model(config, model_type)

    def load_pretrained_embedding(self, modely_type, name):
        return load_pretrained_embedding(self.model_path,modely_type, name).to(self.device_type)
 
class GPT2Components(TransformerComponents):
    def load_components(self):
        self.wte = self.load_pretrained_embedding("gpt2","wte")
        self.wpe = self.load_pretrained_embedding("gpt2","wpe")
        self.lm_head = self.load_pretrained_embedding("gpt2","lm_head")
        self.ln_f = self.load_pretrained_embedding("gpt2","ln_f")
        transformer_components = self.wte, self.wpe,  self.ln_f
        return  transformer_components, self.pretrained_transformer, self.tokenizer, self.lm_head
    

class LlamaComponents(TransformerComponents):
    def load_components(self):
        self.embed_tokens = self.load_pretrained_embedding("llama","embed_tokens")
        self.lm_head = self.load_pretrained_embedding("llama","lm_head")
        self.norm = self.load_pretrained_embedding("llama","norm")
        transformer_components = self.embed_tokens, self.norm
        return transformer_components, self.pretrained_transformer, self.tokenizer, self.lm_head

class MistralComponents(TransformerComponents):
    def load_components(self):
        self.embed_tokens = self.load_pretrained_embedding("mistral","embed_tokens")
        self.lm_head = self.load_pretrained_embedding("mistral","lm_head")
        self.norm = self.load_pretrained_embedding("mistral","norm")
        transformer_components = self.embed_tokens, self.norm
        return transformer_components, self.pretrained_transformer, self.tokenizer, self.lm_head
    

class GptOssComponents(TransformerComponents):
    def load_components(self):
        self.embed_tokens = self.load_pretrained_embedding("gpt_oss","embed_tokens")
        self.lm_head = self.load_pretrained_embedding("gpt_oss","lm_head")
        self.norm = self.load_pretrained_embedding("gpt_oss","norm")
        transformer_components = self.embed_tokens, self.norm
        return transformer_components, self.pretrained_transformer, self.tokenizer, self.lm_head

def load_transformer_components(model_path, device_type, model_type,config):
    if model_type == "gpt2":
        return GPT2Components(model_path, device_type, model_type, config).load_components()
    elif model_type == "llama":
        return LlamaComponents(model_path, device_type, model_type, config).load_components()
    elif model_type == "mistral":
        return MistralComponents(model_path, device_type, model_type, config).load_components()
    elif model_type == "gpt_oss":
        return GptOssComponents(model_path, device_type, model_type, config).load_components()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")