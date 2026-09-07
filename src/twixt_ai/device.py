"""Explicit PyTorch device selection and runtime reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


DeviceRequest = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """Requested and resolved device plus the relevant PyTorch runtime facts."""

    requested_device: DeviceRequest
    resolved_device: Literal["cpu", "cuda"]
    cuda_available: bool
    gpu_name: str | None
    cuda_runtime: str | None
    pytorch_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_device(requested: str) -> DeviceSelection:
    """Resolve ``cpu``, ``cuda``, or deterministic ``auto`` selection.

    ``auto`` chooses CUDA exactly when PyTorch reports it available. An explicit
    CUDA request never falls back to CPU.
    """

    if not isinstance(requested, str):
        raise TypeError("device must be a string")
    if requested not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA was requested but is unavailable to PyTorch")
    resolved = "cuda" if requested == "cuda" or (
        requested == "auto" and cuda_available
    ) else "cpu"
    gpu_name = (
        torch.cuda.get_device_name(torch.device("cuda"))
        if cuda_available
        else None
    )
    return DeviceSelection(
        requested_device=requested,
        resolved_device=resolved,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        cuda_runtime=(
            str(torch.version.cuda) if torch.version.cuda is not None else None
        ),
        pytorch_version=str(torch.__version__),
    )


__all__ = ["DeviceRequest", "DeviceSelection", "select_device"]
