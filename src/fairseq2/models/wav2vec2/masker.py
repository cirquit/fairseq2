# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import override

from fairseq2 import device
from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import InternalError
from fairseq2.nn import BatchLayout
from fairseq2.nn.utils.fairseq1_mask import compute_mask_indices
from fairseq2.nn.utils.mask import RowMaskFactory, compute_row_mask
from fairseq2.typing import get_name_or_self


class Wav2Vec2Masker(Module, ABC):
    """Masks extracted wav2vec 2.0 features."""

    @abstractmethod
    def forward(
        self, seqs: Tensor, batch_layout: BatchLayout | None
    ) -> tuple[Tensor, Tensor]:
        """
        :param seqs:
            The sequences to mask. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param batch_layout:
            The sequence layout information containing batch structure and lengths.

        :returns:
            - The input sequences with mask applied. *Shape:* Same as ``seqs``.
            - The temporal mask that has been applied to ``seqs``. *Shape:*
              :math:`(N,S)`, where :math:`N` is the batch size and :math`S` is
              the sequence length.
        """

    if TYPE_CHECKING:
        __call__ = forward

    @staticmethod
    def extract_masked_elements(seqs: Tensor, temporal_mask: Tensor) -> Tensor:
        """
        Extracts masked elements from ``seqs``.

        :param seqs: The sequences. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param temporal_mask: The temporal mask. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math`S` is the sequence length.
        """
        batch_size = seqs.size(0)

        # (N, S, M) -> (N x T, M)
        seqs = seqs[temporal_mask]

        # (N x T, M) -> (N, T, M)
        return seqs.unflatten(0, (batch_size, -1))  # type: ignore[no-any-return]


@final
class StandardWav2Vec2Masker(Wav2Vec2Masker):
    """Masks extracted wav2vec 2.0 features as described in Section 3.1 of
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`."""

    temporal_mask_embed: Parameter
    temporal_span_len: int
    max_temporal_mask_prob: float

    temporal_mask_embed: Parameter
    min_num_temporal_mask_spans: int
    spatial_span_len: int
    max_spatial_mask_prob: float
    min_num_spatial_mask_spans: int
    mask_factory: RowMaskFactory

    def __init__(
        self,
        mask_codebase: str,
        model_dim: int,
        temporal_span_len: int = 10,
        max_temporal_mask_prob: float = 0.65,
        min_num_temporal_mask_spans: int = 2,
        spatial_span_len: int = 10,
        max_spatial_mask_prob: float = 0.0,
        min_num_spatial_mask_spans: int = 2,
        *,
        mask_factory: RowMaskFactory | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param temporal_span_len:
            The length of each temporal mask span that is applied over time
            steps.
        :param max_temporal_mask_prob:
            The maximum probability of masking a time step. Note that, due to
            mask span overlap, the effective probability will be lower.
        :param spatial_span_len:
            The length of each spatial mask span that is applied over features.
        :param max_spatial_mask_prob:
            The maximum probability of masking a feature. Note that, due to mask
            span overlap, the effective probability will be lower.
        :param mask_factory:
            The row mask factory. If ``None``, :func:`compute_row_mask` will be
            used.
        """
        super().__init__()

        self.mask_factory = mask_factory or compute_row_mask

        if max_temporal_mask_prob == 0.0:
            raise ValueError("`max_temporal_mask_prob` must be greater than 0.")

        self.mask_codebase = mask_codebase
        self.temporal_span_len = temporal_span_len
        self.max_temporal_mask_prob = max_temporal_mask_prob
        self.min_num_temporal_mask_spans = min_num_temporal_mask_spans

        self.temporal_mask_embed = Parameter(
            torch.empty((model_dim,), device=device, dtype=dtype)
        )

        self.spatial_span_len = spatial_span_len
        self.max_spatial_mask_prob = max_spatial_mask_prob
        self.min_num_spatial_mask_spans = min_num_spatial_mask_spans

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        nn.init.uniform_(self.temporal_mask_embed)

    @override
    def forward(
        self, seqs: Tensor, batch_layout: BatchLayout | None
    ) -> tuple[Tensor, Tensor]:

        # TODO: cirquit - implement masking for packed batches
        if batch_layout and batch_layout.packed:
            raise ValueError("`seqs` must not be a packed batch.")

        batch_size, seq_len, model_dim = seqs.shape

        # Temporal mask over time steps.
        if self.mask_codebase == "fairseq2":
            temporal_mask = compute_row_mask(
                shape=(batch_size, seq_len),
                span_len=self.temporal_span_len,
                max_mask_prob=self.max_temporal_mask_prob,
                row_lens=batch_layout.seq_lens_pt if batch_layout is not None else None,
                min_num_spans=self.min_num_temporal_mask_spans,
                device=seqs.device,
            )
        else:
            mask_indices = compute_mask_indices(
                (batch_size, seq_len),
                (
                    None
                    if batch_layout is None
                    else self._create_fairseq1_padding_mask(batch_layout)
                ),
                self.max_temporal_mask_prob,
                self.temporal_span_len,
                mask_type="static",
                mask_other=0.0,
                min_masks=self.min_num_temporal_mask_spans,
                no_overlap=False,
                min_space=1,
                require_same_masks=True,
                mask_dropout=0.0,
            )
            temporal_mask = torch.from_numpy(mask_indices).to(seqs.device)

        if temporal_mask is None:
            raise InternalError("`temporal_mask` is `None`.")

        seqs[temporal_mask] = self.temporal_mask_embed.type_as(seqs)

        if self.max_spatial_mask_prob > 0.0:
            # Spatial mask over features.
            # (N, M)
            spatial_mask = self.mask_factory(
                shape=(batch_size, model_dim),
                span_len=self.spatial_span_len,
                max_mask_prob=self.max_spatial_mask_prob,
                min_num_spans=self.min_num_spatial_mask_spans,
                device=seqs.device,
            )

            if spatial_mask is None:
                raise InternalError("`spatial_mask` is `None`.")

            # (N, M) -> (N, S, M)
            spatial_mask = spatial_mask.unsqueeze(1).expand(-1, seq_len, -1)

            seqs[spatial_mask] = 0.0

        return seqs, temporal_mask

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = (
            f"temporal_span_len={self.temporal_span_len}, "
            f"max_temporal_mask_prob={self.max_temporal_mask_prob}, "
            f"min_num_temporal_mask_spans={self.min_num_temporal_mask_spans}, "
            f"spatial_span_len={self.spatial_span_len}, "
            f"max_spatial_mask_prob={self.max_spatial_mask_prob}, "
            f"min_num_spatial_mask_spans={self.min_num_spatial_mask_spans}"
        )

        if self.mask_factory is not compute_row_mask:
            # TODO:cirquit replaced mask_factory = getattr(self.mask_factory, "__name__", self.mask_factory)
            mask_factory = get_name_or_self(self.mask_factory)

            s = f"{s}, mask_factory={mask_factory}"

        return s

    # TODO: cirquit - write a test for this
    def _create_fairseq1_padding_mask(
        self, batch_layout: BatchLayout | None
    ) -> Optional[Tensor]:
        """
        Creates boolean padding mask for unpacked sequences.
        Valid positions are `False`, invalid positions (padded) are `True`.

        :param batch_layout:
            The sequence layout information containing batch structure and lengths.

        :returns:
            The mask. *Shape:* :math: `(N,S)`, where :math:`N` is the batch size
            and :math:`S` is the maximum sequence length.
        """
        if batch_layout is None:
            return None

        # (N, ) - all sequence lengths
        seq_lens = batch_layout.seq_lens_pt
        batch_size = seq_lens.size(0)
        max_seq_len = batch_layout.max_seq_len

        # (N, S) - 0 to max_seq_len for every sequence
        indices = torch.arange(max_seq_len, device=device).expand(batch_size, -1)

        # (N) -> (N, S) - individual sequence length as every entry in the tensor
        lengths = seq_lens.unsqueeze(1).expand(-1, max_seq_len)

        # (N, S)
        return indices >= lengths
