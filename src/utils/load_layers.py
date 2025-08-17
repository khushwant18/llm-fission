from typing import Dict, Optional, Union
from transformers import PretrainedConfig

from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
import torch

import json
import time
from contextlib import suppress
from typing import Dict, Optional, Union
import safetensors

import torch.nn as nn
# from accelerate import init_empty_weights
# from accelerate.utils import set_module_tensor_to_device
# from hivemind.utils.logging import get_logger
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig
from huggingface_hub import hf_hub_download

from transformers import AutoConfig
from utils.model_type import detect_language_model_family, load_model_block, get_block_prefix, get_embedding_layer

StateDict = Dict[str, torch.Tensor]


def _load_state_dict_from_repo(
    model_name: str,
    block_prefix: str,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
) -> StateDict:
    # if always_needs_auth(model_name) and token is None:
    #     token = True

    index_file = _find_index_file(model_name, revision=revision, token=token, cache_dir=cache_dir)
    if index_file.endswith(".index.json"):  # Sharded model
        path = hf_hub_download(model_name, filename=index_file, token=token, cache_dir=cache_dir, repo_type="model", local_files_only=False)
        if path is None:
            # _find_index_file() told that a file exists but we can't get it (e.g., it just disappeared)
            raise ValueError(f"Failed to get file {index_file}")

        with open(path) as f:
            index = json.load(f)
        filenames = {
            filename for param_name, filename in index["weight_map"].items() if param_name.startswith(block_prefix)
        }
        if not filenames:
            raise RuntimeError(f"Block {block_prefix}* not found in the index: {index['weight_map']}")
    else:  # Non-sharded model
        filenames = {index_file}
    # logger.debug(f"Loading {block_prefix}* from {filenames}")

    state_dict = {}
    for filename in filenames:
        # print("filename",filename,model_name)
        shard_state_dict = _load_state_dict_from_repo_file(
            model_name,
            filename,
            block_prefix=block_prefix,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            max_disk_space=max_disk_space,
        )
        shard_state_dict = {
            param_name[len(block_prefix) :]: param
            for param_name, param in shard_state_dict.items()
            if param_name.startswith(block_prefix)
        }  # Remove unused parameters from memory
        state_dict.update(shard_state_dict)
    return state_dict


INDEX_FILES = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.bin.index.json", "pytorch_model.bin"]


def _find_index_file(
    model_name: str, *, revision: Optional[str] = None, token: Optional[Union[str, bool]] = None, cache_dir: str
) -> str:
    # If we have cached weights (e.g., Pickle from older Petals versions), reuse them
    for filename in INDEX_FILES:
        path = hf_hub_download(
            model_name,
            filename,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            repo_type="model",
            local_files_only=False
        )
        if path is not None:
            return filename

    # If we don't, prefer Safetensors when possible
    # (we don't download files here since we can't account for max_disk_space in case of large files)
    for filename in INDEX_FILES:
        with suppress(EntryNotFoundError):
            get_hf_file_metadata(hf_hub_url(model_name, filename, revision=revision), token=token)
            return filename

    raise ValueError(
        f"Repo {model_name} does not contain weights in a supported format: files {INDEX_FILES} do not exist"
    )


def _load_state_dict_from_repo_file(
    model_name: str,
    filename: str,
    *,
    block_prefix: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
) -> StateDict:
    # First, try to find the weights locally
    try:
        # with allow_cache_reads(cache_dir):
            # print("see",model_name,filename,revision,token,cache_dir)
            path = hf_hub_download(
                model_name,
                filename,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=False
                 repo_type="model"
            )
            # print("path..",path)
            if path is not None:
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
    except Exception:
      print("error")
        # logger.warning(f"Cache for file {filename} is corrupted, it will be downloaded again", exc_info=True)

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    while True:
        try:
            # with allow_cache_writes(cache_dir):
                url = hf_hub_url(model_name, filename, revision=revision)
                file_size = get_hf_file_metadata(url, token=token).size
                # if file_size is not None:
                #     free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                # else:
                #     logger.warning(f"Failed to fetch size of file {filename} from repo {model_name}")

                path = hf_hub_download(
                    model_name,
                    filename,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    repo_type="model"
                )
                if path is None:
                    raise RuntimeError(f"File {filename} does not exist in repo {model_name}")
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
        except Exception as e:
            # logger.warning(f"Failed to load file {filename} from HF Hub (retry in {delay:.0f} sec)", exc_info=True)
            time.sleep(delay)


def _load_state_dict_from_local_file(path: str, *, block_prefix: Optional[str] = None) -> StateDict:
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")

    if path.endswith(".safetensors"):
        # print("path",path)
        # with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        #     tensor_dict = {}
        #     for key in f.keys():
        #         print("key ",key)
        #         if block_prefix is None or key.startswith(block_prefix):
        #             tensor_dict[key] = f.get_tensor(key)
                    
                    

        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys() if block_prefix is None or key.startswith(block_prefix)}

    raise ValueError(f"Unknown weight format: {path}")


def load_pretrained_embedding(
    model_name: str,
    model_type: str,
    emb_type:  str,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    if config is None:
        config = AutoConfig.from_pretrained(model_name)
    # if cache_dir is None:
    #     cache_dir = DEFAULT_CACHE_DIR

    # assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    # torch_dtype = resolve_block_dtype(config, torch_dtype)

    # with init_empty_weights():
    embed_dim = config.hidden_size
    embed_layer, embedding_prefix = get_embedding_layer(config,embed_dim,emb_type,model_type)

   
    state_dict = _load_state_dict_from_repo(
        model_name,
        embedding_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    state_dict = {key[1:]: value for key, value in state_dict.items()}
    # print("state_dict",state_dict.keys())
    # print(state_dict)
    # dummy load, check that keys matchclear

    report = embed_layer.load_state_dict(state_dict, strict=False)

    assert not report.missing_keys, f"Some block weights are missing: {report.missing_keys}"

    for param_name, _ in embed_layer.named_parameters():
        assert param_name in state_dict, f"{param_name} not in state dict"
        param = state_dict[param_name]
        # if not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        #     param = param.to(torch_dtype)
        # set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)

    # logger.info(f"Loaded {model_name} block {block_index}")
    # logger.debug(f"Details: {report}")
    return embed_layer
# print(config.hidden_size)


def load_pretrained_block(
    model_name: str,
    block_index: int,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    if config is None:
        config = AutoConfig.from_pretrained(model_name)
    # if cache_dir is None:
    #     cache_dir = DEFAULT_CACHE_DIR

    # assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    # torch_dtype = resolve_block_dtype(config, torch_dtype)

    # with init_empty_weights():
    model_type = detect_language_model_family(config)
    block = load_model_block(config, model_type,block_index)
    block_prefix = get_block_prefix(block_index, model_type)
    # block_prefix = f"{config.block_prefix}.{block_index}."
    # print("block_prefix",block_prefix)
    
    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    state_dict = {key[1:]: value for key, value in state_dict.items()}
    # print("state_dict",state_dict.keys())
    # print(state_dict)
    # dummy load, check that keys matchclear

    report = block.load_state_dict(state_dict, strict=False)

    assert not report.missing_keys, f"Some block weights are missing: {report.missing_keys}"

    for param_name, _ in block.named_parameters():
        assert param_name in state_dict, f"{param_name} not in state dict"
        param = state_dict[param_name]
        # if not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
        #     param = param.to(torch_dtype)
        # set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)

    # logger.info(f"Loaded {model_name} block {block_index}")
    # logger.debug(f"Details: {report}")
    return block
# print(config.hidden_size)
