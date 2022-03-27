# DOOLY 🦕
PORORO의 아래 단점 세 가지를 개선한 라이브러리입니다.
- 일부 task의 batch화 불가능
- 내부 tokenize 과정 및 모듈 구조 확인이 어려움
- fairseq dependency


## How to use?
아래와 같이 간단하게 사용 가능합니다

- requirements
    - `mecab`, `fugashi`, `ipadic`, `whoosh`
```
$ pip install transformers torch tokenizers dataclasses numpy
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
