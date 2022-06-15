import os
import contextlib
from packaging import version
from typing import Callable, Any

import transformers


DEFAULT_HUB_NAME = "jinmang2/dooly-hub"
DOOLY_HUB_NAME = os.environ.get("DOOLY_HUB_NAME", DEFAULT_HUB_NAME)

hub_utils = [transformers.file_utils]
if version.parse(transformers.__version__) >= version.parse("4.16.0"):
    hub_utils += [getattr(transformers, "utils.hub")]

HF_CO_PREFIX = hub_utils[0].HUGGINGFACE_CO_PREFIX


def register_subfolder(func: Callable) -> Callable:
    BASE_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/"

    def wrapper(*args, **kwargs) -> Any:
        subfolder = kwargs.pop("subfolder", None)
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", DOOLY_HUB_NAME)
        _orig_hf_co_prefixes = []
        if pretrained_model_name_or_path == DOOLY_HUB_NAME:
            for hub_util in hub_utils:
                _orig_hf_co_prefixes.append(hub_util.HUGGINGFACE_CO_PREFIX)
                hub_util.HUGGINGFACE_CO_PREFIX = BASE_PREFIX
                hub_util.HUGGINGFACE_CO_PREFIX += f"{subfolder}/" if subfolder else ""
                hub_util.HUGGINGFACE_CO_PREFIX += "{filename}"
        kwargs.update(dict(
            subfolder=subfolder,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        ))
        output = func(*args, **kwargs)
        for i, hf_co_prefix in enumerate(_orig_hf_co_prefixes):
            hub_utils[i].HUGGINGFACE_CO_PREFIX = hf_co_prefix
        return output

    return wrapper


@contextlib.contextmanager
def recover_original_hf_bucket_url():
    bucket_url_with_subfolder = hub_utils[0].HUGGINGFACE_CO_PREFIX
    for hub_util in hub_utils:
        hub_util.HUGGINGFACE_CO_PREFIX = HF_CO_PREFIX
    yield
    for hub_util in hub_utils:
        hub_util.HUGGINGFACE_CO_PREFIX = bucket_url_with_subfolder


def download_from_hf_hub(
    filename: str, subfolder: str, hub_name: str = None
) -> str:
    hub_name = hub_name or DOOLY_HUB_NAME
    hf_co_resolved_file = hub_utils[0].hf_bucket_url(
        model_id=hub_name, filename=filename, subfolder=subfolder
    )
    resolved_file_path = hub_utils[0].cached_path(hf_co_resolved_file)
    return resolved_file_path