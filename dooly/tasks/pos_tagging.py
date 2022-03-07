from typing import Dict, List, Union

from .base import DoolyTaskConfig
from ..tokenizers import PosTagger, PosTaggerMap


class PosTagging:
    """
    Conduct Part-of-Speech tagging

    Korean (`mecab-ko`)
        - dataset: N/A
        - metric: N/A

    japanese (`mecab-ipadic`)
        - dataset: N/A
        - metric: N/A

    English (`nltk`)
        - dataset: N/A
        - metric: N/A

    Chinese (`jieba`)
        - dataset: N/A
        - metric: N/A

    Args:
        sent (str): input sentence to be tagged

    Returns:
        List[Tuple[str, str]]: list of token and its corresponding pos tag tuple

    """
    task: str = "pos"
    available_langs: List[str] = ["ko", "en", "ja", "zh"]
    available_models: Dict[str, List[str]] = {
        "ko": ["mecab-ko"],
        "en": ["nltk"],
        "ja": ["mecab-ipadic"],
        "zh": ["jieba"],
    }

    def __init__(
        self,
        config: DoolyTaskConfig,
        model: PosTagger
    ):
        self.config = config
        self.model = model

    def __call__(self, sentences: Union[str, List[str]], **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]

        pos_results = []
        for sentence in sentences:
            pos_results.append(self.model.pos(sentence, **kwargs))

        if len(sentences) == 1:
            pos_results = pos_results[0]

        return pos_results

    @classmethod
    def build(
        cls,
        lang: str = None,
        n_model: str = None,
        **kwargs
    ):
        if lang is None:
            lang = cls.available_lang[0]
        if lang not in cls.available_langs:
            raise ValueError
        if n_model is None:
            n_model = cls.available_models[lang][0]
        if n_model not in cls.available_models[lang]:
            raise ValueError

        model = PosTaggerMap[lang]()

        config = DoolyTaskConfig(
            lang=lang,
            n_model=n_model,
            device="cpu",
            misc_tuple=None
        )

        return cls(config=config, model=model)

    def __repr__(self):
        task_info = f"[TASK]: {self.__class__.__name__}"
        category = "[CATEGORY]: SequenceTagging"
        lang_info = f"[LANG]: {self.lang}"
        device_info = "[DEVICE]: cpu"
        model_info = f"[MODEL]: {self.n_model}"
        return "\n".join([task_info, category, lang_info, device_info, model_info])

    @property
    def lang(self):
        return self.config.lang

    @property
    def n_model(self):
        return self.config.n_model
