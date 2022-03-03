import abc
import json
import pickle
import inspect
import warnings
from packaging import version
from functools import partial
from typing import Dict, Union, Optional, Tuple, List, TypeVar

from transformers.file_utils import hf_bucket_url, cached_path


Tokenizer = TypeVar("Tokenizer") # ../tokenizers/base.py


HUB_NAME = "jinmang2/dooly-hub"
VOCAB_NAME = "vocab.json"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

TOKENIZER_USER_AGENT = {"file_type": "tokenizer",
                        "from_auto_class": False,
                        "is_fast": False,}

CONFIG_USER_AGENT = {"file_type": "config",
                     "from_auto_class": False,}

MODEL_USER_AGENT = {"file_type": "model",
                    "framework": "pytorch",
                    "from_auto_class": False,}


def download_from_hf_hub(
    model_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    resume_download: bool = False,
    user_agent: Optional[Union[Dict, str]] = None,
) -> Optional[str]:
    # Resolve a model identifier, a file name, and an optional revision id,
    # to a huggingface.co-hosted url, redirecting to Cloudfront
    # (a Content Delivery Network, or CDN) for large files.
    huggingface_co_resolved_file = hf_bucket_url(
        model_id=model_id,
        filename=filename,
        subfolder=subfolder,
        revision=revision,
    )
    # Given something that might be a URL (or might be a local path),
    # determine which. If it's a URL, download the file and cache it,
    # and return the path to the cached file. If it's already a local path,
    # make sure the file exists and then return the path
    # Do not extract files (extract_compressed_file and force_extract is False)
    resolved_file_path = cached_path(
        huggingface_co_resolved_file,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        user_agent=user_agent,
    )

    return resolved_file_path
