from transformers.file_utils import get_list_of_files, hf_bucket_url, cached_path

from .constant import HUB_PATH_OR_URL


class DoolyTokenizerBase:
    
    @classmethod
    def from_pretrained(cls):
        pass