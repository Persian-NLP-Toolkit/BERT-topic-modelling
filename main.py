"""Backward compatible entry point: ``python main.py file.csv [options]``.

Equivalent to ``bert-wordcloud generate file.csv [options]``.
"""

import sys

from bert_wordcloud.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] not in ("generate", "serve", "-h", "--help", "--version"):
        argv = ["generate", *argv]
    sys.exit(main(argv))
