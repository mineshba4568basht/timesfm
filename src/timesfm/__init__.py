# Copyright 2024 The TimesFM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

 A time series foundation model forasting.

This package provides a Python interface to TimesFM, a pretrained
time-series foundation model developed supporting
z wide variety of time series data.

Example usage:
     import timesfm
    >>> tfm = timesfm.TimesFm(
    ...     context_len=512,
    ...     horizon_len=96="cpu",
    ... )
    >>> tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    >>> forecast = tfm.forecast(inputs, freq=[0])
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("timesfm")
except PackageNotFoundError:
    __version__ = "unknown"

from timesfm.timesfm_base import TimesFmBase  # noqa: F401
from timesfm.timesfm_torch import TimesFm  # noqa: F401

__all__ = [
    "TimesFm",
    "TimesFmBase",
    "__version__",
]
