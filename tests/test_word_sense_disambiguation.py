import os
import json
import pickle
import torch
import unittest
from collections import OrderedDict
from fairseq.file_io import PathManager
from fairseq.models.transformer import TransformerModel
from pororo import Pororo
from pororo.tasks.utils.download_utils import download_or_load

from dooly.hub_interface import HubInterface, TOKENIZER_USER_AGENT
from dooly.models import FSMTConfig, FSMTForConditionalGeneration
from dooly.tasks.word_sense_disambiguation import KoWSDTokenizer


def load_pororo_wsd_model():
    n_model = "transformer.large.ko.wsd"
    model_path = os.path.join("transformer", n_model)

    load_dict = download_or_load(model_path, "ko")
    wsd_model = TransformerModel.from_pretrained(
        model_name_or_path=load_dict.path,
        checkpoint_file=n_model + ".pt",
        data_name_or_path=load_dict.dict_path,
        source_lang=load_dict.src_dict,
        target_lang=load_dict.tgt_dict,
    )

    filename = os.path.join(load_dict.path, n_model) + ".pt"

    if not PathManager.exists(filename):
        raise IOError("Model file not found: {}".format(filename))

    with open(PathManager.get_local_path(filename), "rb") as f:
        state = torch.load(
            f, map_location=lambda s, l: torch.serialization.default_restore_location(s, "cpu")
        )

    def get_model_config(model):

        model_conf = {
            "architectures": ["FSMTForConditionalGeneration"],
            "model_type": "fsmt",
            "activation_dropout": model.args.activation_dropout,
            "activation_function": "relu",
            "attention_dropout": model.args.attention_dropout,
            "d_model": model.args.decoder_embed_dim,
            "dropout": model.args.dropout,
            "init_std": 0.02,
            "max_position_embeddings": model.args.max_source_positions,
            "num_hidden_layers": model.args.encoder_layers,
            "src_vocab_size": len(model.src_dict.indices),
            "tgt_vocab_size": len(model.tgt_dict.indices),
            "langs": [model.args.source_lang, model.args.target_lang],
            "encoder_attention_heads": model.args.encoder_attention_heads,
            "encoder_ffn_dim": model.args.encoder_ffn_embed_dim,
            "encoder_layerdrop": model.args.encoder_layerdrop,
            "encoder_layers": model.args.encoder_layers,
            "encoder_normalize_before": model.args.encoder_normalize_before,
            "decoder_attention_heads": model.args.decoder_attention_heads,
            "decoder_ffn_dim": model.args.decoder_ffn_embed_dim,
            "decoder_layerdrop": model.args.decoder_layerdrop,
            "decoder_layers": model.args.decoder_layers,
            "decoder_normalize_before": model.args.decoder_normalize_before,
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "is_encoder_decoder": True,
            "scale_embedding": not model.args.no_scale_embedding,
            "tie_word_embeddings": model.args.share_all_embeddings,
        }
        model_conf["num_beams"] = 5
        model_conf["early_stopping"] = False
        model_conf["length_penalty"] = 1.0

        return model_conf

    wsd_model_conf = get_model_config(wsd_model)

    with open("config.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(wsd_model_conf, ensure_ascii=False, indent=2))

    return wsd_model


def load_hf_wsd_model(state_dict):
    # rename keys to start with 'model.'
    model_state_dict = OrderedDict(("model." + k, v) for k, v in state_dict.items())
    # remove unneeded keys
    ignore_keys = [
        "model.model",
        "model.encoder.version",
        "model.decoder.version",
        "model.encoder_embed_tokens.weight",
        "model.decoder_embed_tokens.weight",
        "model.encoder.embed_positions._float_tensor",
        "model.decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)

    config = FSMTConfig.from_pretrained("config.json")
    wsd_model_new = FSMTForConditionalGeneration(config)

    # check that it loads ok
    wsd_model_new.load_state_dict(model_state_dict, strict=False)
    wsd_model_new = wsd_model_new.eval()
    return wsd_model_new


class DoolyWordSenseDisambiguationTester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs_wsd_model = load_pororo_wsd_model() # fairseq transformer
        model_state_dict = self.fs_wsd_model.models[0].state_dict()
        self.hf_wsd_model = load_hf_wsd_model(model_state_dict) # huggingface transformer
        HubInterface._build_tokenizer(task="wsd", lang="ko", tokenizer_class=WSDTokenizer)
        self.hf_wsd_tokenizer = WSDTokenizer(vocab)

    def _make_input_tensors(self, text):
        # preprocess
        text = text.replace(" ", "▁")
        text = " ".join([c for c in text])
        # encoding
        input_ids = self.fs_wsd_model.encode(text)
        batches = self.fs_wsd_model._build_batches(input_ids.unsqueeze(0), False)
        for batch in batches:
            break
        eos = self.fs_wsd_model.src_dict.eos_index
        pad = self.fs_wsd_model.src_dict.pad_index
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = (src_tokens.ne(eos) & src_tokens.ne(pad)).long().sum(dim=-1)
        return src_tokens, src_lengths

    def test_tokenizer_encode(self):
        """
        text = "예시 문장입니다. 과연 출력 텐서가 같을까요?"

        def preprocess(text):
            text.replace(" ", "▁")
            text = " ".join([c for c in text])
            return text

        pororo_encoded = self.fs_wsd_model.src_dict.encode_line(
            preprocess(text),
            add_if_not_exist=False
        ).long()

        dooly_encoded = self.hf_wsd_tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=False
        )["input_ids"]

        self.assertTrue(torch.equal(pororo_encoded, dooly_encoded))
        """
        pass

    def test_tokenier_decode(self):
        """
        text = "예시 문장입니다. 과연 출력 텐서가 같을까요?"

        def preprocess(text):
            text.replace(" ", "▁")
            text = " ".join([c for c in text])
            return text

        tensor = self.fs_wsd_model.src_dict.encode_line(
            preprocess(text),
            add_if_not_exist=False
        ).long()

        pororo_decoded = self.fs_wsd_model.string(out)
        dooly_decoded = self.hf_wsd_tokenizer.decode(out[0])

        self.assertTrue(torch.equal(pororo_decoded, dooly_decoded))
        """
        pass

    def test_have_the_same_logits(self):
        # make test sample tensor
        text = "예시 문장입니다. 과연 출력 텐서가 같을까요?"
        src_tokens, src_lengths = self._make_input_tensors(text)

        # calculate logits using pororo(fairseq)
        fs_logits, _ = self.fs_wsd_model.models[0](
            src_tokens,
            src_lengths,
            torch.tensor([[0]])
        )
        # calculate logits using huggingface transformers
        hf_logits = self.hf_wsd_model(
            src_tokens,
            decoder_input_ids=torch.tensor([[0]])
        ).logits

        # check is_equal
        # tensor([[[-1.7683, -1.7684, -1.0957,  ..., -1.7460, -1.7665, -1.7683]]], grad_fn=<UnsafeViewBackward0>)
        self.assertTrue(torch.equal(fs_logits, hf_logits))

    def test_same_inference_result(self):
        """
        추론 결과가 동일한지 검사

        text = "예시 문장입니다. 과연 출력 텐서가 같을까요?"
        _text = text.replace(" ", "▁")
        _text = " ".join([c for c in _text])
        output = wsd_model.translate(_text, beam=5, max_len_a=4, max_len_b=50)
        """
        pass

    def test_output_instance(self):
        """
        출력의 instance가 list인지 출력

        my_results = ...
        self.assertInstance(my_results, list)
        self.assertInstance(my_results, )
        """
        def isnamedtupleinstance(x):
            t = type(x)
            b = t.__bases__
            if len(b) != 1 or b[0] != tuple:
                return False
            f = getattr(t, "_fields", None)
            if not isinstance(f, tuple):
                return False
            return all(type(n) == str for n in f)
        pass

    def test_module_output(self):
        """
        모듈의 출력값이 동일한지 체크

        wsd = Pororo(task="wsd")
        wsd_results = wsd(text)
        my_resulst

        for p, m in zip(wsd_results, my_results):
            self.assertEqual(p, m)

        # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
        """
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
