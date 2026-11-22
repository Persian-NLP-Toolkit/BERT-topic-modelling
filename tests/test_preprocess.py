from bert_wordcloud.preprocess import normalize, parse_stopwords, tokenize


def test_normalize_arabic_chars_and_noise():
    text = "كتاب علي https://example.com @user ۱۲۳ سلامٌ!"
    out = normalize(text)
    assert "ک" in out and "ی" in out
    assert "http" not in out and "user" not in out
    assert "۱" not in out and "ٌ" not in out


def test_tokenize_removes_stopwords_and_short_tokens():
    toks = tokenize("این کتاب و آن دفتر را در خانه دیدم")
    assert toks == ["کتاب", "دفتر", "خانه", "دیدم"]


def test_tokenize_keeps_zwnj_words():
    assert "می‌دهد" in tokenize("او پاسخ می‌دهد")


def test_extra_stopwords():
    assert tokenize("کتاب دفتر", extra_stopwords=["دفتر"]) == ["کتاب"]


def test_parse_stopwords():
    assert parse_stopwords("الف، ب\nپ , ت") == ["الف", "ب", "پ", "ت"]
    assert parse_stopwords(None) == []
