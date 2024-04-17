import argparse
import logging
from flask import Flask, request, jsonify
import torch
from typing import List, Optional
from utils.load_layers import load_pretrained_block
from utils.model_type import detect_language_model_family
from models.llama.custom_modeling_llama import LlamaModel
from transformers import AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    Returns parsed arguments including layer IDs, device type, model repository, and port number.
    """
    parser = argparse.ArgumentParser(description='API for processing model layers.')
    parser.add_argument('--layers', nargs='+', required=True, help='Range of layer IDs (e.g., "1:3").')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu', help='Device type ("cpu", "mps", or "cuda").')
    parser.add_argument('--model', required=True, help='Hugging Face model repository name.')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the API.')
    return parser.parse_args()

def load_blocks(model_path: str, layers: List[str], device_type: str) -> List[torch.nn.Module]:
    """
    Load specified blocks of the model.
    Args:
        model_path: Path to the model.
        layers: List of layer IDs to be loaded.
        device_type: Device type for the model to be loaded onto.
    Returns:
        List of loaded model blocks.
    """
    try:
        blocks = [load_pretrained_block(model_path, b).eval().to(device_type) for b in layers]
    except Exception as e:
        logging.error(f"Failed to load blocks: {e}")
        raise
    return blocks

def process_blocks(blocks: List[torch.nn.Module], hidden_states: torch.Tensor, 
                   model_type: str, cache_position: Optional[torch.Tensor] = None, 
                   position_ids: Optional[torch.Tensor] = None) -> List:
    """
    Process blocks with the given hidden states.
    Args:
        blocks: Loaded model blocks.
        hidden_states: Hidden states tensor.
        model_type: Type of the model (e.g., "gpt2", "llama").
        cache_position: Cache position tensor.
        position_ids: Position IDs tensor.
    Returns:
        Processed hidden states as a list.
    """
    # Process according to the model type
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

    # Example placeholder return, replace with actual processing result
    return hidden_states.to('cpu').detach().numpy().tolist()

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_request():
    """
    Endpoint to process requests.
    Expects hidden_states, and optionally cache_position and position_ids in the request JSON.
    """
    data = request.get_json(force=True)
    hidden_states = torch.tensor(data.get('hidden_states')).to(device_type)

    # Optional fields
    position_ids = torch.tensor(data.get('position_ids')).to(device_type) if data.get('position_ids') else None
    cache_position = torch.tensor(data.get('cache_position')).to(device_type) if data.get('cache_position') else None

    processed_states = process_blocks(blocks, hidden_states, model_type, cache_position, position_ids)
    return jsonify({"res": processed_states})

if __name__ == '__main__':
    args = parse_arguments()

    device_type = args.device
    model_path = args.model
    config = AutoConfig.from_pretrained(model_path)
    model_type = detect_language_model_family(config)

    try:
        start, end = map(int, args.layers[0].split(':'))
        layers = [str(i) for i in range(start, end + 1)]
    except ValueError as e:
        logging.error(f"Invalid layer format: {args.layers[0]}. Expected format 'start:end'. Error: {e}")
        exit(1)
    if model_type == "llama":
        llama=LlamaModel(config) 
    logging.info(f"Deploying layers: {layers}")
    blocks = load_blocks(model_path, layers, device_type)

    app.run(host='0.0.0.0', port=args.port)
