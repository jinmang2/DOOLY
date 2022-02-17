from .modeling_fsmt import FSMTConfig, FSMTForConditionalGeneration
from .modeling_roberta import RobertaConfig, RobertaForSequenceTagging
from ..build import BuildMixin


DoolyModelHub = {
    "ner": {
        "ko": {"charbert.base": RobertaForSequenceTagging},
        "en": {"roberta.base": RobertaForSequenceTagging},
        "ja": {"jaberta.base": RobertaForSequenceTagging},
        "zh": {"zhberta.base": RobertaForSequenceTagging},
    },
    "wsd": {"ko": {"transformer.large": FSMTForConditionalGeneration}},
}
available_tasks = list(DoolyModelHub.keys())


class DoolyModel:

    @classmethod
    def from_pretrained(cls, task: str, lang: str, n_model: str, **kwargs):
        assert task in available_tasks, (
            f"Task `{task}` is not available. See here {available_tasks}."
        )
        available_langs = DoolyModelHub[task]
        assert lang in available_langs, (
            f"Language `{lang}` is not available in this task {task}. "
            f"See here {available_langs}."
        )
        available_models = available_langs[lang]
        assert n_model in available_models, (
            f"Model `{n_model}` is not available in this task-lang pair. "
            f"See here {available_models}."
        )

        model_class = available_models[n_model]

        return BuildMixin._build_model(task, lang, n_model, model_class, **kwargs)
