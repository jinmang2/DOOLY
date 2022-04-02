import os
import json
import shutil
import importlib
import platform
import torch
from collections import OrderedDict
from dataclasses import dataclass
from ..models.modeling_fsmt import FSMTConfig, FSMTForConditionalGeneration
from ..models.modeling_roberta import RobertaConfig


def is_available_pororo():
    return importlib.util.find_spec("pororo")


pf = platform.system()

if pf == "Windows":
    # fairseq 때문에 사실 의미가 없다
    PORORO_SAVE_DIR = "C:\\pororo"
else:
    home_dir = os.path.expanduser("~")
    PORORO_SAVE_DIR = os.path.join(home_dir, ".pororo")


@dataclass
class TaskConfig:
    save_path: str
    task: str
    lang: str
    n_model: str


class DoolyConverter:
    subclasses = {}

    def __init__(self, config: TaskConfig):
        self.config = config
        self._pororo_save_path = PORORO_SAVE_DIR

        if is_available_pororo():
            from pororo import Pororo
        else:
            raise ModuleNotFoundError(
                "Please install pororo with: `pip install pororo`."
            )

        args = dict(task=config.task, lang=config.lang, model=config.n_model)
        self._pororo_model = Pororo(**args)._model
        self._hf_model = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls.subclasses[cls.name] = cls

    @classmethod
    def load(cls, task: str, lang: str, n_model: str, save_path: str, name: str = None):
        converter = cls.subclasses.get(task if name is None else name, None)
        assert converter is not None
        config = TaskConfig(task=task, lang=lang, n_model=n_model, save_path=save_path)
        return converter(config)

    def convert(self):
        # convert vocab
        self.load_and_save_vocab()

        # convert model
        config = self.get_model_config(self._pororo_model)
        hf_model = self.intialize_hf_model(config)

        self._hf_model = self.porting_pororo_to_hf(
            self._pororo_model,
            hf_model,
        )
        self.save_hf_model(self._hf_model)

        # convert misc files
        self.load_and_save_misc()

        return self._pororo_model, self._hf_model

    def check_has_same_logits(self):
        raise NotImplementedError

    @property
    def task(self):
        return self.config.task

    @property
    def lang(self):
        return self.config.lang

    @property
    def n_model(self):
        return self.config.n_model

    @property
    def save_path(self):
        save_path = os.path.join(
            self.config.save_path,
            self.task,
            self.lang,
            ".".join(self.n_model.split(".")[:2]),
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    @property
    def pororo_save_path(self):
        return self._pororo_save_path

    def load_vocab(self):
        raise NotImplementedError

    def save_vocab(self, vocab, filename: str = "vocab.json"):
        with open(os.path.join(self.save_path, filename), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    def load_and_save_vocab(self):
        vocabs = self.load_vocab()
        self.save_vocab(vocabs)

    def get_model_config(self):
        raise NotImplementedError

    def intialize_hf_model(self, config):
        raise NotImplementedError

    def porting_pororo_to_hf(self, pororo_model, hf_model):
        raise NotImplementedError

    def save_hf_model(self, model):
        model.save_pretrained(self.save_path)

    def get_misc_filenames(self):
        raise NotImplementedError

    def load_and_save_misc(self):
        misc_files = self.get_misc_filenames()
        for misc_file in misc_files:
            shutil.move(
                src=os.path.join(self._pororo_save_path, misc_file), dst=self.save_path
            )


class FsmtConverter(DoolyConverter):
    """ Fairseq Machine Translation model Converter """

    name: str = "fsmt"

    def load_vocab(self):
        return self._pororo_model.src_dict.indices

    def get_model_config(self, pororo_model):
        config = {
            "architectures": ["FSMTForConditionalGeneration"],
            "model_type": "fsmt",
            "activation_dropout": pororo_model.args.activation_dropout,
            "activation_function": "relu",
            "attention_dropout": pororo_model.args.attention_dropout,
            "d_model": pororo_model.args.decoder_embed_dim,
            "dropout": pororo_model.args.dropout,
            "init_std": 0.02,
            "max_position_embeddings": pororo_model.args.max_source_positions,
            "num_hidden_layers": pororo_model.args.encoder_layers,
            "src_vocab_size": len(pororo_model.src_dict.indices),
            "tgt_vocab_size": len(pororo_model.tgt_dict.indices),
            "langs": [pororo_model.args.source_lang, pororo_model.args.target_lang],
            "encoder_attention_heads": pororo_model.args.encoder_attention_heads,
            "encoder_ffn_dim": pororo_model.args.encoder_ffn_embed_dim,
            "encoder_layerdrop": pororo_model.args.encoder_layerdrop,
            "encoder_layers": pororo_model.args.encoder_layers,
            "encoder_pre_layernorm": pororo_model.args.encoder_normalize_before,
            "decoder_attention_heads": pororo_model.args.decoder_attention_heads,
            "decoder_ffn_dim": pororo_model.args.decoder_ffn_embed_dim,
            "decoder_layerdrop": pororo_model.args.decoder_layerdrop,
            "decoder_layers": pororo_model.args.decoder_layers,
            "decoder_pre_layernorm": pororo_model.args.decoder_normalize_before,
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "is_encoder_decoder": True,
            "scale_embedding": not pororo_model.args.no_scale_embedding,
            "tie_word_embeddings": pororo_model.args.share_all_embeddings,
        }
        config["num_beams"] = 5
        config["early_stopping"] = False
        config["length_penalty"] = 1.0

        config_path = os.path.join(self.save_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(config, ensure_ascii=False, indent=2))

        return config_path

    def intialize_hf_model(self, config_path):
        config = FSMTConfig.from_pretrained(config_path)
        hf_model = FSMTForConditionalGeneration(config)
        return hf_model

    def porting_pororo_to_hf(self, pororo_model, hf_model):
        state_dict = pororo_model.models[0].state_dict()
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

        # check that it loads ok
        hf_model.load_state_dict(state_dict, strict=False)
        hf_model = hf_model.eval()
        return hf_model


class RobertaConverter(DoolyConverter):
    """ Roberta model Converter """
    
    name: str = "roberta"

    pororo_task_head_name = None
    hf_model_class = None

    def load_vocab(self):
        return self._pororo_model.task.source_dictionary.indices

    def get_model_config(self, pororo_model):
        sent_encoder = pororo_model.model.encoder.sentence_encoder
        config = RobertaConfig(
            vocab_size=sent_encoder.embed_tokens.num_embeddings,
            hidden_size=pororo_model.args.encoder_embed_dim,
            num_hidden_layers=pororo_model.args.encoder_layers,
            num_attention_heads=pororo_model.args.encoder_attention_heads,
            intermediate_size=pororo_model.args.encoder_ffn_embed_dim,
            max_position_embeddings=514,
            type_vocab_size=1,
            layer_norm_eps=1e-5,  # PyTorch default used in fairseq
        )
        cls_heads = pororo_model.model.classification_heads
        task_name = self.pororo_task_head_name
        config.num_labels = cls_heads[task_name].out_proj.weight.shape[0]
        return config

    def intialize_hf_model(self, config):
        hf_model = self.model_class(config)
        hf_model.eval()
        return hf_model

    def porting_pororo_to_hf(self, pororo_model, hf_model):
        sent_encoder = pororo_model.model.encoder.sentence_encoder
        # Now let's copy all the weights.
        # Embeddings
        hf_model.roberta.embeddings.word_embeddings.weight = (
            sent_encoder.embed_tokens.weight
        )
        hf_model.roberta.embeddings.position_embeddings.weight = (
            sent_encoder.embed_positions.weight
        )
        hf_model.roberta.embeddings.token_type_embeddings.weight.data = (
            torch.zeros_like(hf_model.roberta.embeddings.token_type_embeddings.weight)
        )  # just zero them out b/c RoBERTa doesn't use them.
        hf_model.roberta.embeddings.LayerNorm.weight = (
            sent_encoder.emb_layer_norm.weight
        )
        hf_model.roberta.embeddings.LayerNorm.bias = sent_encoder.emb_layer_norm.bias

        for i in range(hf_model.config.num_hidden_layers):
            # Encoder: start of layer
            layer = hf_model.roberta.encoder.layer[i]
            roberta_layer = sent_encoder.layers[i]

            # self attention
            self_attn = layer.attention.self
            assert (
                roberta_layer.self_attn.k_proj.weight.data.shape
                == roberta_layer.self_attn.q_proj.weight.data.shape
                == roberta_layer.self_attn.v_proj.weight.data.shape
                == torch.Size(
                    (hf_model.config.hidden_size, hf_model.config.hidden_size)
                )
            )

            self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
            self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
            self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
            self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
            self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
            self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

            # self-attention output
            self_output = layer.attention.output
            assert (
                self_output.dense.weight.shape
                == roberta_layer.self_attn.out_proj.weight.shape
            )
            self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
            self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
            self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
            self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

            # intermediate
            intermediate = layer.intermediate
            assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
            intermediate.dense.weight = roberta_layer.fc1.weight
            intermediate.dense.bias = roberta_layer.fc1.bias

            # output
            bert_output = layer.output
            assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
            bert_output.dense.weight = roberta_layer.fc2.weight
            bert_output.dense.bias = roberta_layer.fc2.bias
            bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
            bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
            # end of layer

        cls_head = pororo_model.model.classification_heads[self.pororo_task_head_name]
        hf_model.classifier.dense.weight = cls_head.dense.weight
        hf_model.classifier.dense.bias = cls_head.dense.bias
        hf_model.classifier.out_proj.weight = cls_head.out_proj.weight
        hf_model.classifier.out_proj.bias = cls_head.out_proj.bias

        return hf_model
