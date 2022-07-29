# lm_duorat
# Raymond Li, 2020-08-31
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import logging
from typing import List

import numpy as np

import torch
import torch.utils.data

from duorat.models.duorat import DuoRATModel
from duorat.preproc.duorat import duo_rat_decoder_batch, duo_rat_encoder_batch
from duorat.types import (
    RATPreprocItem,
    DuoRATBatch,
    DuoRATDecoderBatch,
)
from duorat.utils import registry

logger = logging.getLogger(__name__)


@registry.register("model", "LMDuoRAT")
class LMDuoRATModel(DuoRATModel):
    def forward(self, preproc_items: List[RATPreprocItem]):

        items = self.preproc_items_to_duorat_items(preproc_items)
        if len(items) == 0:
            return torch.tensor(np.nan)

        encoder_batch = duo_rat_encoder_batch(items=tuple(item.encoder_item for item in items))
        decoder_batch = duo_rat_decoder_batch(items=tuple(item.decoder_item for item in items))

        duo_rat_batch = DuoRATBatch(
            encoder_batch=encoder_batch,
            decoder_batch=decoder_batch,
        )
        memory = self._encode(batch=duo_rat_batch.encoder_batch)
        output = self._decode(memory=memory, batch=duo_rat_batch.decoder_batch)

        assert not torch.isnan(memory).any()
        assert not torch.isnan(output).any()

        loss = self._compute_loss(
            memory=memory,
            output=output,
            target_key_padding_mask=decoder_batch.target_key_padding_mask,
            valid_copy_mask=decoder_batch.valid_copy_mask,
            copy_target_mask=decoder_batch.copy_target_mask,
            valid_actions_mask=decoder_batch.valid_actions_mask,
            target=decoder_batch.target,
        ).mean()

        result = {
            "loss": loss,
            "total": len(preproc_items),
        }
        return result

    @staticmethod
    def _get_targets_as_input(batch: DuoRATDecoderBatch) -> torch.Tensor:
        return batch.shifted_target

    @staticmethod
    def _get_memory_relations(batch: DuoRATDecoderBatch) -> torch.Tensor:
        return batch.shifted_memory_relations
