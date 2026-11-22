import pytest

from bert_wordcloud.ngrams import parse_ngram_range, weight_ngrams
from bert_wordcloud.pipeline import compute_frequencies, load_documents


def test_parse_ngram_range():
    assert parse_ngram_range("1") == (1, 1)
    assert parse_ngram_range("1-2") == (1, 2)
    with pytest.raises(ValueError):
        parse_ngram_range("0-5")


def test_weight_ngrams_counts_and_pmi():
    docs = [["هوش", "مصنوعی"], ["هوش", "مصنوعی"], ["کتاب"]]
    w = weight_ngrams(docs, (1, 2), min_count=2)
    assert w["هوش"] == 2
    assert "هوش مصنوعی" in w and "کتاب" not in w


def test_weight_ngrams_groups_factor():
    docs = [["الف"], ["الف"], ["ب"], ["ب"]]
    w = weight_ngrams(docs, (1, 1), min_count=1, groups=[([0, 1], 3.0), ([2, 3], 1.0)])
    assert w == {"الف": 6.0, "ب": 2.0}


def test_load_csv_default_column(sample_csv):
    docs = load_documents(sample_csv)
    assert docs.column == "txtContent"
    assert len(docs.texts) == 48
    assert docs.columns == ["id", "txtContent"]


def test_load_csv_autodetect_column():
    docs = load_documents("a,body\n1,سلام دنیا\n".encode(), filename="x.csv")
    assert docs.column == "body"


def test_load_txt():
    docs = load_documents("خط اول\n\nخط دوم\n".encode(), filename="x.txt")
    assert docs.texts == ["خط اول", "خط دوم"]


def test_load_rejects_unknown_extension():
    with pytest.raises(ValueError):
        load_documents(b"x", filename="x.pdf")


def test_load_missing_column(sample_csv):
    with pytest.raises(ValueError, match="not found"):
        load_documents(sample_csv, column="nope")


def test_compute_frequencies(sample_csv):
    docs = load_documents(sample_csv)
    res = compute_frequencies(docs.texts, max_words=20)
    terms = dict(res.words)
    assert res.method == "frequency"
    assert "هوش مصنوعی" in terms
    assert "قیمت دلار" in terms
    assert len(res.words) <= 20
    assert res.documents == 48


def test_compute_frequencies_falls_back_when_min_count_too_high():
    res = compute_frequencies(["کتاب خوب"], min_count=5)
    assert res.words and res.warnings


def test_bertopic_falls_back_on_small_input():
    res = compute_frequencies(["کتاب خوب", "کتاب بد"], method="bertopic")
    assert res.method == "frequency"
    assert res.warnings
