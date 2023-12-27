# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import io
import torch
import fsspec
from upath import UPath
from deepspeed.utils import logger, log_dist, timeout_and_retry
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine


class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, tag):
        log_dist(f"[Torch] Checkpoint {tag} is about to be saved!", ranks=[0])

    @timeout_and_retry(
        num_retries=int(os.environ.get("MFSSPEC_NUM_RETRIES", "5")),
        one_retry_timeout=int(os.environ.get("MFSSPEC_TIMEOUT", "1800")),
        finally_skip_error=False,
    )
    def makedirs(self, path, exist_ok=False):
        UPath(path).mkdir(parents=True, exist_ok=exist_ok)

    @timeout_and_retry(
        num_retries=int(os.environ.get("MFSSPEC_NUM_RETRIES", "5")),
        one_retry_timeout=int(os.environ.get("MFSSPEC_TIMEOUT", "1800")),
        finally_skip_error=True,
    )
    def save(self, state_dict, path: str):
        logger.info(f"[Torch] Saving {path}...")
        with fsspec.open(path, "wb") as f:
            torch.save(state_dict, f)
        logger.info(f"[Torch] Saved {path}.")
        return None

    @timeout_and_retry(
        num_retries=int(os.environ.get("MFSSPEC_NUM_RETRIES", "5")),
        one_retry_timeout=int(os.environ.get("MFSSPEC_TIMEOUT", "1800")),
        finally_skip_error=False,
    )
    def load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        with fsspec.open(path, "rb") as f:
            partition = torch.load(f, map_location=map_location)
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True
