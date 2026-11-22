from fastapi.testclient import TestClient

from bert_wordcloud.api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_wordcloud_upload(sample_csv):
    with open(sample_csv, "rb") as fh:
        r = client.post(
            "/api/wordcloud",
            files={"file": ("sample.csv", fh, "text/csv")},
            data={"ngram": "1-2", "max_words": "50", "stopwords": "تیم"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == "txtContent"
    texts = {w["text"] for w in body["words"]}
    assert "هوش مصنوعی" in texts
    assert "تیم" not in texts


def test_wordcloud_bad_file():
    r = client.post(
        "/api/wordcloud", files={"file": ("x.pdf", b"123", "application/pdf")}
    )
    assert r.status_code == 400


def test_wordcloud_bad_ngram(sample_csv):
    with open(sample_csv, "rb") as fh:
        r = client.post(
            "/api/wordcloud", files={"file": ("s.csv", fh)}, data={"ngram": "9"}
        )
    assert r.status_code == 400
