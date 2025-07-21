# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import final

import torch
from torch import Tensor
from torcheval.metrics import MulticlassAccuracy

from fairseq2.datasets import SequenceBatch
from fairseq2.metrics.aggregation import Mean, Sum
from fairseq2.metrics.bag import MetricBag
from fairseq2.model.context import ModelContext
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Loss,
    Wav2Vec2Model,
    Wav2Vec2Output,
    Wav2Vec2VectorQuantizerOutput,
)


@dataclass(kw_only=True)
class Wav2Vec2LossSection:
    diversity_loss_weight: float = 0.1
    """The weight of the diversity loss."""

    feature_penalty_weight: float = 10.0
    """The weight of the regularization penalty applied to the extracted features."""


@final
class Wav2Vec2Criterion:
    _model_context: ModelContext
    _diversity_loss_weight: float
    _feature_penalty_weight: float

    def __init__(
        self,
        model_context: ModelContext,
        diversity_loss_weight: float,
        feature_penalty_weight: float,
    ) -> None:
        if not isinstance(model_context.model.base_module, Wav2Vec2Model):
            raise TypeError(
                f"`model.base_module` must be of type `{Wav2Vec2Model}`, but is of type `{type(model_context.model.base_module)}` instead."
            )

        self._model_context = model_context

        self._diversity_loss_weight = diversity_loss_weight
        self._feature_penalty_weight = feature_penalty_weight

    def __call__(
        self, batch: SequenceBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        output = self._forward(batch)

        loss = output.compute_loss(
            self._diversity_loss_weight, self._feature_penalty_weight
        )

        batch_size, seq_len = output.logits.shape[:2]

        num_targets = batch_size * seq_len

        update_losses(metric_bag, loss, num_targets)

        update_accuracy(metric_bag, output)

        update_quantizer_metrics(metric_bag, output.quantizer_output)

        update_batch_metrics(metric_bag, batch)

        return loss.total, num_targets

    def _forward(self, batch: SequenceBatch) -> Wav2Vec2Output:
        return self._model.module(batch)  # type: ignore[no-any-return]

    # @property
    # def model(self) -> Model:
    #    return self._model


@torch.inference_mode()
def update_losses(metric_bag: MetricBag, loss: Wav2Vec2Loss, num_targets: int) -> None:
    n = num_targets

    d = num_targets * math.log(2)

    metric_bag.get(Mean, "loss").update(loss.total.detach() / d, weight=n)

    metric_bag.get(Mean, "contrastive_loss").update(
        loss.contrastive.detach() / d, weight=n
    )

    metric_bag.get(Mean, "diversity_loss").update(loss.diversity.detach() / d, weight=n)

    metric_bag.get(Mean, "feature_penalty").update(
        loss.feature_penalty.detach() / d, weight=n
    )


@torch.inference_mode()
def update_accuracy(metric_bag: MetricBag, output: Wav2Vec2Output) -> None:
    # (N x S)
    predictions = output.logits.argmax(-1).view(-1)

    # wav2vec2 treats logit at index 0 as the target.
    targets = torch.zeros_like(predictions)

    metric_bag.get(MulticlassAccuracy, "accuracy").update(predictions, targets)


@torch.inference_mode()
def update_quantizer_metrics(
    metric_bag: MetricBag, output: Wav2Vec2VectorQuantizerOutput
) -> None:
    if not isinstance(output, Wav2Vec2VectorQuantizerOutput):
        return

    # TODO: cirquit - this output class does not have these parameters, dig up when they were deleted
    # metric_bag.get(Mean, "code_perplexity").update(output.code_perplexity)
    # metric_bag.get(Mean, "prob_perplexity").update(output.prob_perplexity)
    # metric_bag.get(Mean, "temperature").update(output.temperature)


@torch.inference_mode()
def update_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    """Update the batch metrics."""
    num_examples = batch.batch_size

    num_elements = batch.num_elements()

    metric_bag.get(Sum, "num_examples").update(num_examples)
    metric_bag.get(Sum, "num_elements").update(num_elements)


# TODO: cirquit - check whether this is required as we don't have this state anymore and would need to pipe it from the criterion
# if self._train:
#     assert self.total_num_examples is not None
#     assert self.total_num_elements is not None

#     self.total_num_examples.update(num_examples)
#     self.total_num_elements.update(num_elements)
