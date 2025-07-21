# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed

from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    AdamWConfig,
    CommonSection,
    DatasetSectionBase,
    GangSection,
    LRSchedulerSection,
    ModelSection,
    OptimizerSection,
    RegimeSection,
    TrainerSection,
)

from .dataset import GENERIC_SPEECH_DATASET_FAMILY


@dataclass(kw_only=True)
class Wav2Vec2TrainConfig:
    """
    The default values correspond to the base ls960h training setup as described
    in :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    model: ModelSection = field(
        default_factory=lambda: ModelSection(family="wav2vec2", arch="base")
    )

    dataset: Wav2Vec2TrainDatasetSection = field(
        default_factory=lambda: Wav2Vec2TrainDatasetSection()
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: TrainerSection = field(
        default_factory=lambda: TrainerSection(dtype=torch.float16)
    )

    loss: Wav2Vec2LossSection = field(default_factory=lambda: Wav2Vec2LossSection())

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5e-04, betas=(0.9, 0.98), eps=1e-06, weight_decay=0.01
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=POLYNOMIAL_DECAY_LR,
            config=PolynomialDecayLRConfig(num_warmup_steps=32_000),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=400_000,
            validate_every_n_steps=5_000,
            checkpoint_every_n_steps=25_000,
            publish_metrics_every_n_steps=200,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())


@dataclass(kw_only=True)
class Wav2Vec2TrainDatasetSection(DatasetSectionBase):
    name: str | None = "librispeech_960h"
    """The name, path or path to the asset card of the speech dataset."""

    family: str = GENERIC_SPEECH_DATASET_FAMILY

    path: Path | None = None

    train_split: str = "train"
    """The name of the train data split."""

    valid_split: str | None = "valid"
    """The name of the valid data split."""

    min_audio_len: int = 32_000
    """The minimum audio sequence length."""

    max_audio_len: int = 250_000
    """The maximum audio sequence length."""

    max_num_elements: int = 1_500_000
    """The maximum number of elements per batch."""

    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    use_fbank: bool = False

    no_padding: bool = True

    example_shuffle_window: int = 500_000
    """The size of the sliding window for shuffling examples."""

    batch_shuffle_window: int = 0
    """The size of the sliding window for shuffling batches."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""

    npc: int = 10
    """ TODO: cirquit - find out what that is"""

    extras: dict[str, object] = field(default_factory=dict)
    """The dataset-specific extra options."""
