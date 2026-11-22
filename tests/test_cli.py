import json

from bert_wordcloud.cli import main


def test_cli_generate_png_and_json(sample_csv, tmp_path):
    png = tmp_path / "out.png"
    js = tmp_path / "out.json"
    code = main(
        ["generate", str(sample_csv), "-o", str(png), "--json", str(js), "--top", "5"]
    )
    assert code == 0
    assert png.stat().st_size > 1000
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["words"]


def test_cli_missing_file(tmp_path):
    assert main(["generate", str(tmp_path / "missing.csv"), "-o", "-"]) == 2
