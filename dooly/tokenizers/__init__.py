from .base import Tokenizer
from .bpe import Gpt2BpeTokenizer, BpeJaZhTokenizer
from .char import CharS1Tokenizer, CharS2Tokenizer
from ..build import BuildMixin


DoolyTokenizerHub = {
    "ner": {
        "ko": {"charbert.base": CharS1Tokenizer},
        "en": {"roberta.base": Gpt2BpeTokenizer},
        "ja": {"jaberta.base": BpeJaZhTokenizer},
        "zh": {"zhberta.base": BpeJaZhTokenizer},
    },
    "wsd": {
        "ko": {"transformer.large": CharS2Tokenizer},
    },
}
available_tasks = list(DoolyTokenizerHub.keys())


class DoolyTokenizer(Tokenizer):

    @classmethod
    def from_pretrained(cls, task: str, lang: str, n_model: str, **kwargs):
        assert task in available_tasks, (
            f"Task `{task}` is not available. See here {available_tasks}."
        )
        available_langs = DoolyTokenizerHub[task]
        assert lang in available_langs, (
            f"Language `{lang}` is not available in this task {task}. "
            f"See here {available_langs}."
        )
        available_models = available_langs[lang]
        assert n_model in available_models, (
            f"Model `{n_model}` is not available in this task-lang pair. "
            f"See here {available_models}."
        )

        tokenizer_class = available_models[n_model]

        return BuildMixin._build_tokenizer(task, lang, n_model, tokenizer_class, **kwargs)
