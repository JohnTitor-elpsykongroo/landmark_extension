"""Runtime compatibility for old type-only annotations on newer PyTorch."""

import sys
import types

import torch


def ensure_torchtyping() -> None:
    try:
        import torchtyping  # noqa: F401
    except RuntimeError as exc:
        if "Cannot subclass _TensorBase directly" not in str(exc):
            raise

        class TensorType:
            def __class_getitem__(cls, _shape):
                return torch.Tensor

        module = types.ModuleType("torchtyping")
        module.TensorType = TensorType
        sys.modules["torchtyping"] = module
