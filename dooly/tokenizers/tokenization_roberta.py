from typing import List, Union, Optional
from tokenizers import Encoding
from transformers import RobertaTokenizerFast as _RobertaTokenizerFast

from .base import InputTexts, TokenizedOutput


def convert_vocab_from_fairseq_to_hf(vocab_path):
    import json
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    new_vocab = {"<pad>": 1, "<mask>": len(vocab) + 1}
    for v, i in vocab.items():
        if v == "<s>":
            new_vocab[v] = 0
        elif v == "</s>":
            new_vocab[v] = 2
        elif v == "<unk>":
            new_vocab[v] = 3
        else:
            new_vocab[v] = i + 1
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(new_vocab, f, ensure_ascii=False)


def build_custom_roberta_tokenizer(
    name_or_path: str,
    vocab_filename: str,
    merges_filename: Optional[str] = None,
    add_prefix_space: bool = True,
    replacement="▁",
):
    import tokenizers

    bpe_obj = tokenizers.models.BPE.from_file(
        vocab_filename,
        merges_filename,
        unk_token="<unk>",
        fuse_unk=True,
    )
    # @TODO: Unigram
    _tokenizer = tokenizers.Tokenzizer(bpe_obj)
    _tokenizer.normalizer = tokenizers.normalizers.NFKC()
    _tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace(
        replacement=replacement,
        add_prefix_space=add_prefix_space,
    )
    _tokenizer.post_processor = tokenizers.processors.RobertaProcessing(
        sep=("</s>", 2),
        cls=("<s>", 0),
        add_prefix_space=False,
    )
    _tokenizer.decoder = tokenizers.decoders.Metaspace(
        replacemment=replacement,
        add_prefix_space=add_prefix_space,
    )

    return RobertaTokenizerFast(
        tokenizer_object=_tokenizer,
        add_prefix_space=add_prefix_space,
        name_or_path=name_or_path,
    )


# To match the class name to avoid warning statements
# when `config_tokenizer_class` is not None.
# See here: https://github.com/huggingface/transformers/blob/133c5e40c4c34b54180f1f0f48791bece45f4418/src/transformers/tokenization_utils_base.py#L1825
class RobertaTokenizerFast(_RobertaTokenizerFast):

    def segment(self, texts: InputTexts) -> TokenizedOutput:
        encodings = self._tokenizer.encode_batch(
            texts,
            add_special_tokens=False,
        )
        results = []
        for encoding in encodings:
            results.append(self._unk_to_raw_text(encoding))
        return results

    def _unk_to_raw_text(self, encoding: Encoding) -> List[str]:
        offsets = encoding.offsets
        tokens = encoding.tokens
        result = []
        for offset, token in zip(offsets, tokens):
            if token != "<unk>":
                result.append(token)
                continue
            s, e = offset
            result.append(text[s:e])
        return result
