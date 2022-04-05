import os
import importlib


def is_available_mecab():
    _mecab = importlib.util.find_spec("mecab")
    if os.name != "nt":
        return _mecab is not None
    else:
        _eunjeon = importlib.util.find_spec("eunjeon")
        return _mecab is not None and _eunjeon is not None


def is_available_ipadic():
    return importlib.util.find_spec("ipadic")


def is_available_fugashi():
    return importlib.util.find_spec("fugashi")


def is_available_jieba():
    return importlib.util.find_spec("jieba")


def is_available_nltk():
    return importlib.util.find_spec("nltk")


def is_available_kss():
    return importlib.util.find_spec("kss")
