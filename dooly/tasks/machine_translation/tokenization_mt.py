import os
from typing import List, Union, Dict

import torch

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize as en_sent_tokenize

from kss import split_sentences

from tokenizers import Tokenizer
from tokenizers import decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def sent_tokenize(texts: Union[str, List[str]],
                  langs: Union[str, List[str]]) -> List[List[str]]:
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(langs, str):
        langs = [langs] * len(texts)
    if len(set(langs)) == 1 and langs[0] == "ko":
        sentences = split_sentences(texts)
    else:
        sentences = []
        for text, lang in zip(texts, langs):
            if lang == "en":
                sentences.append(en_sent_tokenize(text))
            elif lang == "ko":
                sentences.append(split_sentences(text))
            else: # lang in ["ja", "zh"]
                text = text.replace("。", "。[SEP]")
                text = text.replace("！", "！[SEP]")
                text = text.replace("？", "？[SEP]")
                if "[SEP]" in text:
                    sents = text.split("[SEP]")
                    sents = sents[:-1]
                else:
                    sents = [text]
                sentences.append(sents)
    num_sentences = [len(sents) for sents in sentences]
    return sentences, num_sentences


def _langtok(lang: str):
    mapping = {"en": "_XX", "ja": "_XX", "ko": "_KR", "zh": "_CN"}
    return f"[{lang + mapping[lang]}]"


_tokenizer = Tokenizer(
    BPE.from_file(
        vocab=os.path.join(load_dict.src_tok, "vocab.json"),
        merges=os.path.join(load_dict.src_tok, "merges.txt"),
        unk_token="<unk>",
        fuse_unk=True,
    )
)
_tokenizer.normalizer = NFKC()
_tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
    replacement="▁",
    add_prefix_space=True,
)
_tokenizer.decoder = decoders.Metaspace(
    replacement="▁",
    add_prefix_space=True,
)


tokenized_results = []
for sents, src, tgt in zip(sentences, src_langs, tgt_langs):
    if src == "en":
        encodings = _tokenizer.encode_batch(sents)
        results = []
        for text, encoding in zip(sents, encodings):
            offsets = encoding.offsets
            tokens = encoding.tokens
            res = []
            for offset, token in zip(offsets, tokens):
                if token != "<unk>":
                    res.append(token)
                    continue
                s, e = offset
                res.append(text[s:e])
            pieces = " ".join(res)
            results.append(f"{_langtok(src)} {pieces} {_langtok(tgt)}")
    else: # src in ["ko", "ja", "zh"]
        results = []
        for text in sents:
            pieces = " ".join([c if c != " " else "▁" for c in text.strip()])
            results.append(f"{_langtok(src)} {pieces} {_langtok(tgt)}")
    tokenized_results.append(results)
    
    
def encode(
    sentence: text, 
    vocab: Dict[str, int], 
    append_eos: bool = True, 
    eos_token_id: int = 2, 
    unk_token_id: int = 3
):
    words = tokenize_line(sentence)
    nwords = len(words)
    ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
    for i, word in enumerate(words):
        idx = vocab.get(word, unk_token_id)
        ids[i] = idx
    if append_eos:
        ids[nwords] = eos_token_id
    return ids