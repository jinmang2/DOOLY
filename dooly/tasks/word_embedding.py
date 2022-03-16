from typing import List, Dict, Tuple, Union, Optional
from collections import OrderedDict

from .base import DoolyTaskConfig, Miscellaneous
from .base.wikipedia2vec import (
    QueryParser,
    WhooshIndex,
    Wikipedia2Vec,
)


class WordEmbedding(Miscellaneous):
    """
    Get vector or find similar word and entity from pretrained model using wikipedia

    See also:
        Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia (https://arxiv.org/abs/1812.06280)

    Korean (`wikipedia2vec.ko`)
        - dataset: kowiki-20200720
        - metric: N/A

    English (`wikipedia2vec.en`)
        - dataset: enwiki-20180420
        - metric: N/A

    Japanese (`wikipedia2vec.ja`)
        - dataset: jawiki-20180420
        - metric: N/A

    Chinese (`wikipedia2vec.zh`)
        - dataset: zhwiki-20180420
        - metric: N/A

    Args:
        query (str): input qeury
        top_n (int): number of result word or entity (need for `find_similar_words`)
        group (bool): return grouped dictionary or not (need for `find_similar_words`)

    Notes:
        Wikipedia2Vec has two diffrent kinds of output format following below.
        1. 'something' (word) : word2vec result (non-hyperlink in wikipedia documents)
        2. 'something' (other) : entity2vec result (hyperlink in wikipedia documents)

    """
    task: str = "word_embedding"
    available_langs: List[str] = ["ko", "en", "ja", "zh"]
    available_models: Dict[str, List[str]] = {
        "ko": ["wikipedia2vec.ko"],
        "en": ["wikipedia2vec.en"],
        "ja": ["wikipedia2vec.ja"],
        "zh": ["wikipedia2vec.zh"],
    }
    misc_files: Dict[str, List[str]] = {
        "ko": ["kowiki_20200720_100d.pkl", "ko_indexdir.zip"],
        "en": ["enwiki_20180420_100d.pkl", "en_indexdir.zip"],
        "ja": ["jawiki_20180420_100d.pkl", "ja_indexdir.zip"],
        "zh": ["zhwiki_20180420_100d.pkl", "zh_indexdir.zip"],
    }

    def __init__(self, config: DoolyTaskConfig):
        super().__init__(config=config)
        self._model = Wikipedia2Vec(self.misc_tuple[0], self.device)
        self._ix = WhooshIndex.open_dir(self.misc_tuple[1])
