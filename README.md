# DOOLY 🦕
PORORO는 kakaobrain에서 개발한 라이브러리로 nlp를 공부하는 이들에게 큰 도움이 되었습니다. 하지만 PORORO에는 아래 세 가지 단점이 존재합니다.
- 일부 task의 batch화 불가능
- 내부 tokenize 과정 및 모듈 구조 확인이 어려움
- fairseq dependency

Dooly는 위 단점 세 가지를 개선한 라이브러리입니다.
- 모든 task를 batch화하여 inference 가능
- task별 tokenizer, model을 모듈로 분리하여 출력값 확인 가능
- 모든 것을 huggingface transformers로 처리

## How to use?
아래와 같이 간단하게 사용 가능합니다

- requirements (v0.1.1 -> setup으로 해결)
    - `mecab`, `fugashi`, `ipadic`, `whoosh`, `nltk`
```
$ pip install transformers datasets torch tokenizers dataclasses numpy
```

- install

```
$ pip install dooly
```

- how to use
    - PORORO와 동일하게 사용할 수 있습니다.
```python
from dooly import Dooly

ner = Dooly(task="ner", lang="ko")
```

## Support Tasks
- Back Translation Data Augmentation
- Dependency Parsing
- Machine Reading Comprehension
- Machine Translation
- Named Entity Recognition
- Natural Language Inference
- Pos Tagging
- Question Generation
- Word Embedding
- Word Sense Disambiguation
- Zero Shot Topic Classification



## Reference
- https://github.com/kakaobrain/pororo
