"""Render frequencies to a PNG image (used by the CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

FONT_PATH = Path(__file__).parent / "fonts" / "Gandom.ttf"

#: Default colours, the same "vivid" palette as the web UI.
VIVID = (
    "#e63946",
    "#2a9d8f",
    "#f4a261",
    "#3a86ff",
    "#8338ec",
    "#ff006e",
    "#264653",
    "#e9a800",
)

RGB = Tuple[int, int, int]


def _shape(text: str) -> str:
    """Reshape + reorder Persian text so Pillow renders it correctly."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


def _luminance(rgb: RGB) -> float:
    def lin(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a: RGB, b: RGB) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def readable(color: RGB, background: RGB, minimum: float = 3.0) -> RGB:
    """Mix ``color`` towards black/white until it is readable on ``background``."""
    if _contrast(color, background) >= minimum:
        return color
    target = (255, 255, 255) if _luminance(background) < 0.4 else (0, 0, 0)
    for step in range(1, 11):
        t = step / 10
        mixed = tuple(round(c + (tc - c) * t) for c, tc in zip(color, target))
        if _contrast(mixed, background) >= minimum:  # type: ignore[arg-type]
            return mixed  # type: ignore[return-value]
    return target


def _palette(colormap: Optional[str], n: int) -> Sequence[RGB]:
    from PIL import ImageColor

    if not colormap:
        return [ImageColor.getrgb(c)[:3] for c in VIVID]
    import matplotlib

    cmap = matplotlib.colormaps[colormap]
    count = getattr(cmap, "N", 256)
    # Qualitative maps (tab10, Set2 ...) have few colours: use them all.
    # Continuous maps: sample evenly, skipping the extreme (often near-white) ends.
    if count <= 20:
        samples = [cmap(i) for i in range(count)]
    else:
        k = max(2, min(n, 12))
        samples = [cmap(0.15 + 0.7 * i / (k - 1)) for i in range(k)]
    return [tuple(round(v * 255) for v in s[:3]) for s in samples]  # type: ignore[misc]


def render_png(
    frequencies: Dict[str, float],
    output: Union[str, Path],
    width: int = 1600,
    height: int = 900,
    background: str = "white",
    colormap: Optional[str] = None,
    font_path: Optional[Union[str, Path]] = None,
    transparent: bool = False,
    scale: float = 2,
    prefer_horizontal: float = 1.0,
) -> Path:
    """Draw a word cloud.

    The layout is computed at ``width`` x ``height`` and drawn ``scale`` times
    larger, so the default produces a sharp 3200x1800 image.
    """
    from PIL import ImageColor
    from wordcloud import WordCloud

    if not frequencies:
        raise ValueError("Nothing to render: the frequency table is empty.")
    shaped: Dict[str, float] = {}
    for text, weight in frequencies.items():
        key = _shape(text)
        shaped[key] = shaped.get(key, 0.0) + weight

    bg_rgb: RGB = (
        (255, 255, 255) if transparent else ImageColor.getrgb(background)[:3]  # type: ignore[assignment]
    )
    colors = [readable(c, bg_rgb) for c in _palette(colormap, len(shaped))]
    rank = {
        word: i
        for i, word in enumerate(sorted(shaped, key=shaped.__getitem__, reverse=True))
    }

    def color_func(word: str, **_kwargs) -> str:
        r, g, b = colors[rank.get(word, 0) % len(colors)]
        return f"rgb({r}, {g}, {b})"

    wc = WordCloud(
        width=width,
        height=height,
        scale=scale,
        background_color=None if transparent else background,
        mode="RGBA" if transparent else "RGB",
        color_func=color_func,
        max_words=len(shaped),
        # Keep the biggest word from swallowing the image and the smallest readable.
        max_font_size=max(24, int(min(width, height) * 0.22)),
        min_font_size=max(10, int(min(width, height) * 0.016)),
        relative_scaling=0.45,
        font_step=1,
        margin=4,
        prefer_horizontal=prefer_horizontal,
        font_path=str(font_path or FONT_PATH),
        random_state=42,
    )
    wc.generate_from_frequencies(shaped)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wc.to_image().save(out, optimize=True)
    return out
