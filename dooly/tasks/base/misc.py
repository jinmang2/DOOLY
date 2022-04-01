from .base import DoolyTaskBase


class Miscellaneous(DoolyTaskBase):
    @classmethod
    def build(cls, *args, **kwargs):
        if cls.__name__ == "WordEmbedding":
            _ = kwargs.pop("device", None)
            kwargs.update({"device": None})
        return super().build(*args, **kwargs)
