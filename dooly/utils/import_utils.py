# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright 2022 The HuggingFace Team. All rights reserved.
import os
import importlib
from typing import Any


def is_available_mecab() -> bool:
    _mecab = importlib.util.find_spec("mecab")
    if os.name != "nt":
        return _mecab is not None
    else:
        _eunjeon = importlib.util.find_spec("eunjeon")
        return _mecab is not None or _eunjeon is not None


def is_available_ipadic() -> bool:
    return importlib.util.find_spec("ipadic")


def is_available_fugashi() -> bool:
    return importlib.util.find_spec("fugashi")


def is_available_jieba() -> bool:
    return importlib.util.find_spec("jieba")


def is_available_nltk() -> bool:
    return importlib.util.find_spec("nltk")


def is_available_kss() -> bool:
    return importlib.util.find_spec("kss")


def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj