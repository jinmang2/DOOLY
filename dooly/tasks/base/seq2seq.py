from typing import Union, List, Tuple, Dict, Optional

import torch

from .base import DoolyTaskBase


class Seq2Seq(DoolyTaskBase):

    @torch.no_grad()
    def generate(
        self,
        text: Union[List[str], str],
        src_lang: Union[List[str], str] = None,
        tgt_lang: Union[List[str], str] = None,
        add_special_tokens: bool = True,
        no_separator: bool = False,
        beams: int = 5,
        max_len_a: int = 4,
        max_len_b: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
        **kwargs,
    ):
        # Tokenize and get input_ids
        inputs, = self._preprocess(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )
        input_ids = self._prepare_inputs(inputs)["input_ids"]

        do_sample = False
        if top_k or top_p:
            assert isinstance(top_p, float) and 0 <= top_p <= 1
            do_sample = True

            if top_k is not None:
                assert isinstance(top_k, int) and top_k > 0

            if top_p is not None:
                assert isinstance(top_p, float) and 0 <= top_p <= 1

        # Do not support beam_sample
        if do_sample and beams > 1:
            beams = 1

        # @TODO: fix miss-match fairseq vs transformers
        outputs = self.model.generate(
            input_ids,
            num_beams=beams,
            max_length=input_ids.shape[-1] * max_len_a + max_len_b,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            **kwargs,
        )

        return outputs
