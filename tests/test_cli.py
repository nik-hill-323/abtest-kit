import json

import pandas as pd
import pytest

from abtest.cli import main


def test_power_cli(capsys):
    main(["power", "--baseline", "0.1", "--mde", "0.02"])
    out = json.loads(capsys.readouterr().out)
    assert 3700 <= out["n_per_arm"] <= 3900


def test_ztest_cli(capsys):
    main(["ztest", "--control", "480", "1000", "--treatment", "530", "1000"])
    out = json.loads(capsys.readouterr().out)
    assert out["significant"] is True


def test_ttest_cli_from_csv(tmp_path, capsys):
    df = pd.DataFrame(
        {"group": ["control"] * 50 + ["treatment"] * 50, "metric": list(range(50)) + list(range(5, 55))}
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    main(["ttest", "--csv", str(path)])
    out = json.loads(capsys.readouterr().out)
    assert out["difference"] == 5.0


def test_mde_means_cli(capsys):
    main(["mde-means", "--std", "12", "--n-per-arm", "8000"])
    out = json.loads(capsys.readouterr().out)
    assert out["mde"] == pytest.approx(0.532, abs=0.005)


def test_mde_means_cli_round_trips_power_means(capsys):
    main(["power-means", "--std", "10", "--mde", "2"])
    n = json.loads(capsys.readouterr().out)["n_per_arm"]
    main(["mde-means", "--std", "10", "--n-per-arm", str(n)])
    assert json.loads(capsys.readouterr().out)["mde"] == pytest.approx(2, rel=1e-3)


def test_ratio_cli_from_csv(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "group": ["control"] * 3 + ["treatment"] * 3,
            "revenue": [10.0, 20.0, 30.0, 12.0, 24.0, 36.0],
            "sessions": [1, 2, 3, 1, 2, 3],
        }
    )
    path = tmp_path / "users.csv"
    df.to_csv(path, index=False)
    main(["ratio", "--csv", str(path), "--numerator-col", "revenue", "--denominator-col", "sessions"])
    out = json.loads(capsys.readouterr().out)
    assert out["control_ratio"] == pytest.approx(10.0)
    assert out["treatment_ratio"] == pytest.approx(12.0)
    assert out["relative_lift"] == pytest.approx(0.2)
    assert out["control_n_users"] == 3
