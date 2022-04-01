import collections
from typing import Union, List, Tuple, Dict

import torch
import numpy as np
from transformers.modeling_outputs import ModelOutput

from .base import batchify, DoolyTaskWithModelTokenzier


class SequenceTagging(DoolyTaskWithModelTokenzier):
    def find_nbest_predictions(
        self,
        examples: List[str],
        features: Dict[str, Union[torch.Tensor, List]],
        predictions: ModelOutput,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        version2_with_negative: bool = False,
    ):
        # Build a map example to its corresponding features
        all_start_logits, all_end_logits = predictions[:2]
        if isinstance(all_start_logits, torch.Tensor):
            all_start_logits = all_start_logits.detach().cpu().numpy()
        if isinstance(all_end_logits, torch.Tensor):
            all_end_logits = all_end_logits.detach().cpu().numpy()

        features_per_example = collections.defaultdict(list)
        for i, example_id in enumerate(features["example_id"]):
            features_per_example[example_id].append(i)

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict() if version2_with_negative else None

        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # Those are the indices of the features associated to the current example
            feature_indices = features_per_example[example_index]

            min_null_predictions = None
            prelim_predictions = []

            # Looping through all the features associated to the current example
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # score_null = s1 + e1
                feature_null_score = start_logits[0] + end_logits[0]
                # This is what will allow us to map some the positions
                # in our logits to span of texts in the original context
                offset_mapping = features["offset_mapping"][feature_index]

                # Update minimum null prediction
                if (
                    min_null_predictions is None
                    or min_null_predictions["score"] > feature_null_score
                ):
                    min_null_predictions = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibles for the n_best_size greater start end end logits
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1  # noqa
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1  # noqa
                ].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers!
                        # either because the indices are out of bounds
                        # or correspond to part of the input_ids that are note in the context
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length negative or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": (
                                    start_logits[start_index] + end_logits[end_index]
                                ),
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )

            if version2_with_negative:
                # Add the minimum null predictions
                prelim_predictions.append(min_null_predictions)
                null_score = min_null_predictions["score"]

            # Only keep the best `n_best_size` predictions
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score
            if version2_with_negative and not any(
                p["offsets"] == (0, 0) for p in predictions
            ):
                predictions.append(min_null_predictions)

            # Use the offsets to gather the answer text in the original context
            context = example
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["offsets"] = offsets
                pred["text"] = context[offsets[0] : offsets[1]]  # noqa

            # In the very rare edge case we have not a single non-null prediction
            # we create a fake predictions to avoid failure
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):
                predictions.insert(
                    0,
                    dict(
                        text="",
                        offsets=(0, 0),
                        start_logit=0.0,
                        end_logit=0.0,
                        score=0.0,
                    ),
                )

            # Compute the softmax of all scores
            # (we do it with numpy to stay independent from torch in this file,
            # using the LogSum trick)
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction
            # If the null answer is not possible, this is easy
            if not version2_with_negative:
                all_predictions[example_index] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction
                i = 0
                try:
                    while predictions[i]["text"] == "":
                        i += 1
                except IndexError:
                    i = 0
                best_non_null_pred = predictions[i]

                # Then we compare to the null predictions using the threshold
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example_index] = float(
                    score_diff
                )  # To be JSON-serializable
                if score_diff > null_score_diff_threshold:
                    all_predictions[example_index] = ""
                else:
                    all_predictions[example_index] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float
            all_nbest_json[example_index] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        return all_predictions, all_nbest_json, scores_diff_json

    @batchify("question", "context")
    @torch.no_grad()
    def predict_span(
        self,
        question: List[str],
        context: List[str],
        n_best_size: int = 20,
        null_score_diff_threshold: float = 0.0,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:
        # Tokenize and get input_ids
        (inputs,) = self._preprocess(
            text=question,
            text_pair=context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )

        # Predict span
        model_inputs = self._remove_unused_columns(inputs)
        if isinstance(model_inputs["input_ids"], list):
            model_inputs = self.tokenizer.pad(
                {"input_ids": model_inputs["input_ids"]},
                return_tensors="pt",
            )
        model_inputs = self._prepare_inputs(model_inputs)
        predictions = self.model(**model_inputs)

        predictions, all_nbest, scores_diff = self.find_nbest_predictions(
            examples=context,
            features=inputs,
            predictions=predictions,
            n_best_size=n_best_size,
            null_score_diff_threshold=null_score_diff_threshold,
        )

        # dict to list
        predictions = list(predictions.values())
        all_nbest = list(all_nbest.values())
        if scores_diff is not None:
            scores_diff = list(scores_diff.values())

        return predictions, all_nbest, scores_diff

    @batchify("sentences")
    @torch.no_grad()
    def predict_tags(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True,  # ENBERTa, JaBERTa, ZhBERTa에선 없음
        no_separator: bool = False,
    ):
        # Tokenize and get input_ids
        inputs, tokens = self._preprocess(
            sentences,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
            return_tokens=True,
        )

        if len(sentences) == 1:
            tokens = [tokens]

        # Predict tags and ignore <s> & </s> tokens
        inputs = self._prepare_inputs(inputs)
        logits = self.model(**inputs).logits
        if add_special_tokens:
            logits = logits[:, 1:-1, :]
        results = logits.argmax(dim=-1).cpu().numpy()

        # Label mapping
        labelmap = lambda x: self.model.config.id2label[x]  # noqa
        labels = np.vectorize(labelmap)(results)

        token_label_pairs = [
            [(tok, l) for tok, l in zip(sent, label)]
            for sent, label in zip(tokens, labels)
        ]

        return token_label_pairs

    @batchify("sentences")
    @torch.no_grad()
    def predict_dependency(
        self,
        sentences: Union[List[str], str],
        add_special_tokens: bool = True,
    ) -> Tuple:
        # Tokenize and get input_ids
        inputs, tokens_with_pair = self._preprocess(
            sentences,
            add_special_tokens=add_special_tokens,
            return_tokens=True,
            return_tags=True,
        )
        tokens = tokens_with_pair[0]

        if len(sentences) == 1:
            tokens = [tokens]

        attention_mask = inputs["attention_mask"]
        sent_lengths = attention_mask.sum(-1).detach().cpu().numpy() - 2

        inputs = self._prepare_inputs(inputs)
        dp_outputs = self.model(**inputs)

        heads = dp_outputs.classifier_attention
        labels = dp_outputs.logits

        heads = heads.argmax(dim=-1).detach().cpu().numpy()[:, 1:-1]
        labels = labels.argmax(dim=-1).detach().cpu().numpy()[:, 1:-1]

        # out-of-index handling
        labelmap0 = np.vectorize(
            lambda x: self._label0.get(x + self.tokenizer.nspecial, "-1")
        )
        labelmap1 = np.vectorize(
            lambda x: self._label1.get(x + self.tokenizer.nspecial, "-1")
        )

        return tokens, labelmap0(heads), labelmap1(labels), sent_lengths
