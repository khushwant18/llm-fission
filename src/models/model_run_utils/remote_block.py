import numpy as np
import requests
import torch
# from ...client import device_type
def process_hidden_states(layer_url_map, hidden_states, device_type,cache_position=None,position_ids=None,batch_size=None,seq_length=None):
    """
    Processes the given hidden states by sending them to the specified URLs and returns the modified hidden states.
    
    Args:
        1. layer_url_map (list[str]): List of URL strings to send HTTP POST requests to.
        2. hidden_states (torch.Tensor): The hidden states to be processed.
    
    Returns:
        torch.Tensor: Modified hidden states after processing.
    """
    # Convert the hidden states tensor to a CPU Numpy array
    current_hidden_states_np = hidden_states.detach().cpu().numpy().tolist()
    if cache_position != None:
        cache_position = cache_position.detach().cpu().numpy().tolist()
    if position_ids != None:
        position_ids = position_ids.detach().cpu().numpy().tolist()
   
    for url in layer_url_map:
        try:
            # Send an HTTP POST request with the current hidden states data
            response = requests.post(url, json={"hidden_states": current_hidden_states_np,"cache_position":cache_position,"position_ids":position_ids,"batch_size":batch_size,"seq_length":seq_length})

            if response.status_code == 200:
                # Extract the 'res' field from the response JSON
                res = response.json().get('res', None)

                if res is not None:
                    # Update the current hidden states with the received data
                    current_hidden_states_np = res
            else:
                print(f"Warning! Received non-200 status code '{response.status_code}' when posting to {url}")
        except Exception as e:
            print(f"Error occurred while making a request to {url}: {e}")

    # Convert the final hidden states back to a Pytorch tensor
    final_hidden_states = torch.tensor(current_hidden_states_np).to(device_type)
    return final_hidden_states