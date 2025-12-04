from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from thermompnn_d import v2_ssm
from thermompnn import train_thermompnn


@pytest.mark.usefixtures("datadir")
def test_thermompnn_cli_single_writes_csv(monkeypatch, datadir, tmp_path):
    captured = {}

    def fake_get_config(mode):
        captured["config_mode"] = mode
        return SimpleNamespace(model=SimpleNamespace(num_final_layers=1, edges=False, mutant_embedding=False))

    def fake_get_model(mode, cfg):
        captured["model_args"] = (mode, cfg)
        return object()

    def fake_load_pdb(path, chains):
        captured["pdb_path"] = path
        captured["chains"] = chains
        return {"seq": "AC"}

    def fake_run_single_ssm(pdb_data, cfg, model):
        captured["run_single_called"] = True
        return "ddg_tensor", "sequence_tensor"

    def fake_format_output_single(ddg, S, threshold):
        captured["format_args"] = (ddg, S, threshold)
        return [-1.23], ["A1V"]

    def fake_renumber_pdb(df, pdb_data, mode):
        captured["renumber_mode"] = mode
        return df

    monkeypatch.setattr(v2_ssm, "get_config", fake_get_config)
    monkeypatch.setattr(v2_ssm, "get_model", fake_get_model)
    monkeypatch.setattr(v2_ssm, "load_pdb", fake_load_pdb)
    monkeypatch.setattr(v2_ssm, "run_single_ssm", fake_run_single_ssm)
    monkeypatch.setattr(v2_ssm, "format_output_single", fake_format_output_single)
    monkeypatch.setattr(v2_ssm, "renumber_pdb", fake_renumber_pdb)

    out_base = tmp_path / "results"
    exit_code = v2_ssm.main([
        "--mode",
        "single",
        "--pdb",
        str(datadir / "dummy.pdb"),
        "--out",
        str(out_base),
    ])

    assert exit_code == 0

    csv_path = Path(f"{out_base}.csv")
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert list(df["ddG (kcal/mol)"]) == [-1.23]
    assert list(df["Mutation"]) == ["A1V"]

    assert captured["config_mode"] == "single"
    assert captured["pdb_path"] == str(datadir / "dummy.pdb")
    assert captured["format_args"][2] == -0.5
    assert captured["renumber_mode"] == "single"


def test_train_thermompnn_cli_invokes_training(monkeypatch, shared_datadir):
    captured = {}

    class DummyDataset:
        def __init__(self, name):
            self.name = name

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"item": index}

    train_dataset = DummyDataset("train")
    val_dataset = DummyDataset("val")

    def fake_get_v2_dataset(cfg):
        captured["dataset_cfg"] = cfg
        return train_dataset, val_dataset

    class FakeDataLoader:
        def __init__(self, dataset, *args, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def __iter__(self):
            return iter([])

    class FakeModel:
        def __init__(self, cfg):
            captured.setdefault("model_cfgs", []).append(cfg)

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_init"] = kwargs

        def fit(self, model, train_loader, val_loader):
            captured["fit_args"] = (model, train_loader, val_loader)

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            captured["checkpoint_kwargs"] = kwargs

    class FakeLogger:
        def __init__(self, **kwargs):
            captured["logger_kwargs"] = kwargs

    def fake_wandb_init(*args, **kwargs):
        captured["wandb_kwargs"] = kwargs
        return SimpleNamespace()

    def fake_makedirs(path, exist_ok=False):
        captured["makedirs_path"] = path

    monkeypatch.setattr(train_thermompnn, "get_v2_dataset", fake_get_v2_dataset)
    monkeypatch.setattr(train_thermompnn, "DataLoader", FakeDataLoader)
    monkeypatch.setattr(train_thermompnn, "TransferModelPLv2", FakeModel)
    monkeypatch.setattr(train_thermompnn, "TransferModelPLv2Siamese", FakeModel)
    monkeypatch.setattr(train_thermompnn.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_thermompnn, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(train_thermompnn, "WandbLogger", FakeLogger)
    monkeypatch.setattr(train_thermompnn.wandb, "init", fake_wandb_init)
    monkeypatch.setattr(train_thermompnn.os, "makedirs", fake_makedirs)

    cfg_paths = [shared_datadir / "base.yaml", shared_datadir / "override.yaml"]
    exit_code = train_thermompnn.main([str(cfg_paths[0]), str(cfg_paths[1])])

    assert exit_code == 0

    trainer_kwargs = captured["trainer_init"]
    assert trainer_kwargs["max_epochs"] == 3
    assert trainer_kwargs["accelerator"] == "cpu"
    assert trainer_kwargs["limit_train_batches"] == pytest.approx(0.25)
    assert trainer_kwargs["log_every_n_steps"] == 100

    fit_model, fit_train_loader, fit_val_loader = captured["fit_args"]
    assert isinstance(fit_model, FakeModel)
    assert fit_train_loader.dataset is train_dataset
    assert fit_val_loader.dataset is val_dataset
    assert fit_train_loader.kwargs["batch_size"] == 2

    assert captured["wandb_kwargs"] == {"project": "test-project", "name": "cli-run"}
    assert captured["dataset_cfg"].training.epochs == 3
    assert "makedirs_path" in captured