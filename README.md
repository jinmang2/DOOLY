# DOOLY 🦕
PORORO의 아래 단점 세 가지를 개선한 라이브러리입니다.
- 일부 task의 batch화 불가능
- 내부 tokenize 과정 및 모듈 구조 확인이 어려움
- fairseq dependency

아래의 PORORO에서 지원하지 않는 task도 모델 학습 후 배포할 예정입니다.
- information retrieval
- dialogue state tracking

## How to use?
- requirements
    - `mecab`, `fugashi`, `ipadic`
```
$ pip install transformers torch tokenizers dataclasses numpy
```

- install
    - pip install도 가능하게 할 예정입니다
    - task 10개 이상 추가되면 tag 배포하겠습니다.
```
$ git clone https://github.com/jinmang2/DOOLY.git
$ cd DOOLY
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

### TODO
- SRL(NER처럼)
- STS, Sentence Embedding
- PG, Summarization
- Retrieval(DPR)
- convert + tester
- github action - test_xxx.py


## Versions
- v0.x.x의 목적: 일단 pororo의 모든 모듈을 dooly로 사용 가능하게 만든다
- v1.x.x의 목적
    - hf datasets, pipeline을 활용하여 코드 가독성과 효율성을 늘린다
    - `len(...) == 1` 이런 코드 제거하기
- v2.x.x: 계획 없음

## Reference
- https://github.com/kakaobrain/pororo
