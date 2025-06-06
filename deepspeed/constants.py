# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from datetime import timedelta

#############################################
# Torch distributed constants
#############################################
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500

# Default process group wide timeout, if applicable.
# This only applies to the gloo and nccl backends
# (only if NCCL_BLOCKING_WAIT or NCCL_ASYNC_ERROR_HANDLING is set to 1).
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout = timedelta(minutes=int(os.getenv("DEEPSPEED_TIMEOUT", default=30)))
INFERENCE_GENERIC_MODE = 'generic'
INFERENCE_SPECIALIZED_MODE = 'specialized'

CROSS_RANK = "CROSS_RANK"
CROSS_SIZE = "CROSS_SIZE"
LOCAL_RANK = 'LOCAL_RANK'
