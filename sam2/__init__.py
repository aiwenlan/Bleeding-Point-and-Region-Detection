# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
import os

# 确保Hydra能够找到SAM2的配置文件
if not GlobalHydra.instance().is_initialized():
    # 初始化配置模块
    initialize_config_module("sam2", version_base="1.2")
