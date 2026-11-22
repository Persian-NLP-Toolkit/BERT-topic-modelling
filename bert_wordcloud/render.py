"""Render frequencies to a PNG image (used by the CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

FONT_PATH = Path(__file__).parent / "fonts" / "Gandom.ttf"


def _shape(text: str) -> str:
    """Reshape + reorder Persian text so Pillow renders it correctly."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


def render_png(
    frequencies: Dict[str, float],
    output: Union[str, Path],
    width: int = 1600,
    height: int = 900,
    background: str = "white",
    colormap: str = "tab20",
    font_path: Optional[Union[str, Path]] = None,
    transparent: bool = False,
) -> Path:
    from wordcloud import WordCloud

    if not frequencies:
        raise ValueError("Nothing to render: the frequency table is empty.")
    shaped: Dict[str, float] = {}
    for text, weight in frequencies.items():
        key = _shape(text)
        shaped[key] = shaped.get(key, 0.0) + weight

    wc = WordCloud(
        width=width,
        height=height,
        background_color=None if transparent else background,
        mode="RGBA" if transparent else "RGB",
        colormap=colormap,
        max_words=len(shaped),
        min_font_size=10,
        prefer_horizontal=0.9,
        font_path=str(font_path or FONT_PATH),
        random_state=42,
    )
    wc.generate_from_frequencies(shaped)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(out))
    return out
