from typing import List, Dict, Tuple, Union, Callable, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import DoolyTaskConfig, Seq2Seq
from ..tokenizers import Tokenizer as _Tokenizer
from ..tokenizers import RobertaTokenizerFast
from ..build_utils import HUB_NAME

Tokenizer = Union[_Tokenizer, PreTrainedTokenizerBase]


class MachineTranslation(Seq2Seq):
    """
    Machine translation using Transformer models

    Multi (`transformer.large.multi.mtpg`)
        - dataset: Train (Internal data) / Test (Multilingual TED Talk)
        - metric: BLEU score
            +-----------------+-----------------+------------+
            | Source Language | Target Language | BLEU score |
            +=================+=================+============+
            | Average         |  X              |   10.00    |
            +-----------------+-----------------+------------+
            | English         |  Korean         |   15       |
            +-----------------+-----------------+------------+
            | English         |  Japanese       |   8        |
            +-----------------+-----------------+------------+
            | English         |  Chinese        |   8        |
            +-----------------+-----------------+------------+
            | Korean          |  English        |   15       |
            +-----------------+-----------------+------------+
            | Korean          |  Japanese       |   10       |
            +-----------------+-----------------+------------+
            | Korean          |  Chinese        |   4        |
            +-----------------+-----------------+------------+
            | Japanese        |  English        |   11       |
            +-----------------+-----------------+------------+
            | Japanese        |  Korean         |   13       |
            +-----------------+-----------------+------------+
            | Japanese        |  Chinese        |   4        |
            +-----------------+-----------------+------------+
            | Chinese         |  English        |   16       |
            +-----------------+-----------------+------------+
            | Chinese         |  Korean         |   10       |
            +-----------------+-----------------+------------+
            | Chinese         |  Japanese       |   6        |
            +-----------------+-----------------+------------+
        - ref: http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/
        - note: This result is about out of domain settings, TED Talk data wasn't used during model training.

    Multi (`transformer.large.multi.fast.mtpg`)
        - dataset: Train (Internal data) / Test (Multilingual TED Talk)
        - metric: BLEU score
            +-----------------+-----------------+------------+
            | Source Language | Target Language | BLEU score |
            +=================+=================+============+
            | Average         |  X              |   8.75     |
            +-----------------+-----------------+------------+
            | English         |  Korean         |   13       |
            +-----------------+-----------------+------------+
            | English         |  Japanese       |   6        |
            +-----------------+-----------------+------------+
            | English         |  Chinese        |   7        |
            +-----------------+-----------------+------------+
            | Korean          |  English        |   15       |
            +-----------------+-----------------+------------+
            | Korean          |  Japanese       |   11       |
            +-----------------+-----------------+------------+
            | Korean          |  Chinese        |   10       |
            +-----------------+-----------------+------------+
            | Japanese        |  English        |   3        |
            +-----------------+-----------------+------------+
            | Japanese        |  Korean         |   13       |
            +-----------------+-----------------+------------+
            | Japanese        |  Chinese        |   4        |
            +-----------------+-----------------+------------+
            | Chinese         |  English        |   15       |
            +-----------------+-----------------+------------+
            | Chinese         |  Korean         |   8        |
            +-----------------+-----------------+------------+
            | Chinese         |  Japanese       |   4        |
            +-----------------+-----------------+------------+
        - ref: http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/
        - note: This result is about out of domain settings, TED Talk data wasn't used during model training.

    Args:
        text (str): input text to be translated
        beam (int): beam search size
        temperature (float): temperature scale
        top_k (int): top-K sampling vocabulary size
        top_p (float): top-p sampling ratio
        no_repeat_ngram_size (int): no repeat ngram size
        len_penalty (float): length penalty ratio

    Returns:
        str: machine translated sentence

    """
    task: str = "mt"
    available_langs: List[str] = ["multi"]
    available_models: Dict[str, List[str]] = {
        "multi": [
            "transformer.large.mtpg",
            "transformer.large.fast.mtpg",
        ],
    }
    misc_files: Dict[str, List[str]] = {}

    def __init__(
        self,
        config: DoolyTaskConfig,
        tokenizer: Tokenizer,
        model: PreTrainedModel,
    ):
        super().__init__(config=config)
        self._tokenizer = tokenizer
        self._model = model

        # set sentence tokenizer
        self._tokenizer._set_sent_tokenizer()

        # set langtok
        if "mtpg" in self.n_model:
            self._tokenizer.langtok_style = "mbart"
        elif "m2m" in self.n_model:
            self._tokenizer.langtok_style = "multilingual"
        else:
            self._tokenizer.langtok_style = "basic"

        # set sub-tokenizer
        for lang in ["ko", "en", "ja", "zh"]:
            try:
                subtok = RobertaTokenizerFast.from_pretrained(
                    pretrained_model_name_or_path=HUB_NAME,
                    subfolder=f"{self.task}/{self.lang}/{self.n_model}/{lang}",
                )
                self._tokenizer._set_sub_tokenizer(lang, subtok)
            except:
                continue

        self.finalize()

    def __call__(
        self,
        sentences: Union[List[str], str],
        src_langs: Union[List[str], str],
        tgt_langs: Union[List[str], str],
        add_special_tokens: bool = True,
        do_sent_split: bool = True,
        beams: int = 5,
        max_len_a: int = 1,
        max_len_b: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        no_repeat_ngram_size: int = 4,
        length_penalty: float = 1.0,
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]

        if do_sent_split:
            sentences, n_sents = self.tokenizer.sent_tokenize(sentences)
            sentences = [sentence for sents in sentences for sentence in sents]
            src_langs = [lang for lang, n in zip(src_langs, n_sents) for i in range(n)]
            tgt_langs = [lang for lang, n in zip(tgt_langs, n_sents) for i in range(n)]
        else:
            n_sents = [1] * len(sentences)

        generated = self.generate(
            text=sentences,
            src_lang=src_langs,
            tgt_lang=tgt_langs,
            add_special_tokens=add_special_tokens,
            no_separator=False,
            beams=beams,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            **kwargs
        )

        decoded_text = self.tokenizer.decode(generated)

        results = []
        ix = 0
        for n_sent in n_sents:
            result = ""
            for _ in range(n_sent):
                result += decoded_text[ix] + " "
                ix += 1
            results.append(result[:-1])

        if len(results) == 1:
            results = results[0]

        return results
