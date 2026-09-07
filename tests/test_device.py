from __future__ import annotations

import json

import pytest

from twixt_ai import device as device_module
from twixt_ai.device import select_device
from twixt_ai.device_cli import main


def test_cpu_selection_records_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    selection = select_device("cpu")

    assert selection.requested_device == "cpu"
    assert selection.resolved_device == "cpu"
    assert selection.cuda_available is False
    assert selection.gpu_name is None
    assert selection.pytorch_version == str(device_module.torch.__version__)


def test_auto_selects_mocked_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_module.torch.cuda, "get_device_name", lambda _: "Mock RTX")

    selection = select_device("auto")

    assert selection.requested_device == "auto"
    assert selection.resolved_device == "cuda"
    assert selection.cuda_available is True
    assert selection.gpu_name == "Mock RTX"


def test_explicit_cuda_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="requested but is unavailable"):
        select_device("cuda")


@pytest.mark.parametrize("requested", ["mps", "cuda:0", ""])
def test_rejects_devices_outside_runtime_contract(requested: str) -> None:
    with pytest.raises(ValueError, match="cpu.*cuda.*auto"):
        select_device(requested)


def test_probe_emits_machine_readable_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--device", "cpu"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["requested_device"] == "cpu"
    assert output["resolved_device"] == "cpu"
