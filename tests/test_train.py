import sys

import pytest

import train


@pytest.fixture
def fake_yolo(monkeypatch):
    """Replace train.YOLO with a stub that records train() kwargs."""
    recorded = {}

    class FakeYOLO:
        def __init__(self, model):
            recorded["model"] = model

        def train(self, **kwargs):
            recorded["train_kwargs"] = kwargs

    monkeypatch.setattr(train, "YOLO", FakeYOLO)
    return recorded


def test_parse_args_exposes_new_attributes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train.parse_args()
    assert args.optimizer is None
    assert args.lr0 is None
    assert args.no_amp is False


def test_parse_args_parses_new_values(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--optimizer", "AdamW", "--lr0", "0.001", "--no-amp"],
    )
    args = train.parse_args()
    assert args.optimizer == "AdamW"
    assert args.lr0 == pytest.approx(0.001)
    assert args.no_amp is True


def test_main_passes_new_kwargs_when_set(monkeypatch, fake_yolo):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--optimizer", "AdamW", "--lr0", "0.001", "--no-amp"],
    )
    train.main()
    kwargs = fake_yolo["train_kwargs"]
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == pytest.approx(0.001)
    assert kwargs["amp"] is False


def test_main_omits_new_kwargs_when_unset(monkeypatch, fake_yolo):
    """Default (old-style) commands must keep 100% identical train() kwargs."""
    monkeypatch.setattr(sys, "argv", ["train.py"])
    train.main()
    kwargs = fake_yolo["train_kwargs"]
    assert "optimizer" not in kwargs
    assert "lr0" not in kwargs
    assert "amp" not in kwargs
