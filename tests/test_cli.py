import json

import pandas as pd

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
