#!/usr/bin/env python3
"""
glitch.py — Cyber Glitch Effect: генератор глитча для фотографий.

Повторяет приём с референсов: монохромная тонировка, полосы со срезами,
растяжкой, progressive/motion blur, грязные шумные линии и россыпь глифов.

Примеры:
    python3 glitch.py photo.jpg -o out/ -n 6 --seed 42
    python3 glitch.py photo.jpg -o out/ --tint blue --dirt 1.4
    python3 glitch.py photo.jpg -o out/ --tint '#ff5a1f' --text off

Один и тот же seed + вход = один и тот же результат.
"""

from __future__ import annotations

import argparse
import colorsys
import io
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------
# Палитра
# ----------------------------------------------------------------------------

PRESETS = {
    "blue": (0.57, 0.92, 0.95),      # кобальт
    "orange": (0.055, 0.95, 0.96),   # коралловый
    "green": (0.21, 0.95, 0.83),     # кислотный лайм
    "magenta": (0.92, 0.90, 0.94),   # фуксия
    "amber": (0.09, 0.90, 0.97),
    "cyan": (0.51, 0.90, 0.93),
    "violet": (0.75, 0.80, 0.90),
}

# Коэффициент растушёвки краёв: 0 — рубленые, 1 — как сейчас, 2 — совсем мягкие.
FEATHER = 1.0


def _feather_width(value: float) -> int:
    return max(0, int(round(value * FEATHER)))


# Набор символов с референсов: цифры, решётка, слэш и несколько букв.
CHARSET = "0000112344568#/#/XHMSSI=_" + "ABCDEFHKLMNPRSTUVWXYZ"


def hex_to_rgb(value: str) -> np.ndarray:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def hsv_hex(h: float, s: float, v: float) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return np.array([r, g, b], dtype=np.float32)


# ----------------------------------------------------------------------------
# Базовые операции
# ----------------------------------------------------------------------------

def load_image(path: str, width: int | None = None, height: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if width and height:
        im = cover_resize(im, width, height)
    elif width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def cover_resize(im: Image.Image, w: int, h: int) -> Image.Image:
    """Кадрирование под точный размер без искажения пропорций."""
    scale = max(w / im.width, h / im.height)
    nw, nh = round(im.width * scale), round(im.height * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - w) // 2
    top = (nh - h) // 2
    return im.crop((left, top, left + w, top + h))


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def to_float(im: Image.Image) -> np.ndarray:
    return np.asarray(im).astype(np.float32) / 255.0


def resize_arr(arr: np.ndarray, w: int, h: int, method: int = Image.BILINEAR) -> np.ndarray:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.resize((max(1, w), max(1, h)), method)).astype(np.float32) / 255.0


def luminance(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def mix(a, b, t: float):
    return a * (1 - t) + b * t


# ----------------------------------------------------------------------------
# Тонировка
# ----------------------------------------------------------------------------

def dominant_hue(arr: np.ndarray) -> float:
    """Преобладающий оттенок по насыщенным пикселям — чтобы тинт шёл от снимка."""
    small = arr[::8, ::8].reshape(-1, 3)
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in small], dtype=np.float32)
    weight = hsv[:, 1] * hsv[:, 2]
    mask = (hsv[:, 1] > 0.35) & (hsv[:, 2] > 0.18)
    if mask.sum() < 40:
        return 0.57
    hist, edges = np.histogram(hsv[mask, 0], bins=72, range=(0, 1), weights=weight[mask])
    i = int(hist.argmax())
    lo, hi = max(0, i - 2), min(72, i + 3)
    idx = np.arange(lo, hi)
    centre = (idx * hist[lo:hi]).sum() / max(hist[lo:hi].sum(), 1e-6)
    return (centre + 0.5) / 72.0


def resolve_tint(name: str | None, src: np.ndarray, rng) -> np.ndarray:
    if name and name.strip().lower() in ("none", "no", "original", "off"):
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if not name or name == "auto":
        h = dominant_hue(src)
        return hsv_hex(h, 0.92, 0.95)
    key = name.strip().lower()
    if key in PRESETS:
        return hsv_hex(*PRESETS[key])
    if key.startswith("#") or (len(key) in (3, 6) and all(c in "0123456789abcdef" for c in key)):
        return hex_to_rgb(key)
    return hsv_hex(*PRESETS["blue"])


def neutral_grade(img: np.ndarray, contrast: float = 1.05,
                  saturation: float = 1.06) -> np.ndarray:
    """Без тонировки: исходные цвета, только лёгкий контраст и насыщенность."""
    lum = luminance(img)[..., None]
    out = np.clip(lum + (img - lum) * saturation, 0, 1)
    out = np.clip(0.5 + (out - 0.5) * contrast, 0, 1)
    return out



def _blur_arr(arr: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.4:
        return arr
    return to_float(to_pil(arr).filter(ImageFilter.GaussianBlur(float(radius))))


def progressive_areas(img: np.ndarray, rng, strength: float = 1.0) -> np.ndarray:
    """
    Случайные участки с прогрессивным размытием: резкий край переходит в мягкий.
    Накладывается только на часть кадра, а не на всё изображение.
    """
    if strength <= 0.02:
        return img
    H, W, _ = img.shape
    out = img.copy()
    n = int(np.clip(round(rng.integers(2, 4) * strength), 1, 6))

    for _ in range(n):
        h = int(rng.integers(int(H * 0.05), max(int(H * 0.06), int(H * 0.22))))
        y = int(rng.integers(0, max(1, H - h)))
        w = int(rng.integers(int(W * 0.25), max(int(W * 0.26), int(W * 0.85))))
        x0 = int(rng.integers(0, max(1, W - w)))
        w = min(w, W - x0)
        if h < 6 or w < 12:
            continue

        patch = out[y:y + h, x0:x0 + w]
        radius = max(1.0, min(h, w) * rng.uniform(0.02, 0.07) * strength)
        blurred = _blur_arr(patch, radius)

        axis = 0 if rng.random() < 0.5 else 1
        ramp = np.linspace(0.0, 1.0, patch.shape[axis], dtype=np.float32) ** rng.uniform(0.8, 1.6)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        wgt = ramp[:, None, None] if axis == 0 else ramp[None, :, None]

        # мягкие края участка, чтобы не было прямоугольника
        edge = np.ones((h, w), dtype=np.float32)
        f = int(min(14, h // 3, w // 3))
        if f > 1:
            r1 = np.linspace(0.0, 1.0, f, dtype=np.float32)
            edge[:f] *= r1[:, None]
            edge[-f:] *= r1[::-1][:, None]
            edge[:, :f] *= r1[None, :]
            edge[:, -f:] *= r1[::-1][None, :]
        wgt = wgt * edge[..., None]

        out[y:y + h, x0:x0 + w] = np.clip(patch * (1 - wgt) + blurred * wgt, 0, 1)

    return out


def duotone(img: np.ndarray, tint: np.ndarray, preserve: float = 0.12,
            contrast: float = 1.08, dirt: float = 1.0) -> np.ndarray:
    """Сжимает картинку в один оттенок: тени уходят в комплементарный сумрак."""
    lum = luminance(img)
    lum = np.clip((lum - 0.015) / 0.985, 0, 1) ** (1.02 + 0.05 * dirt)
    lum = np.clip(0.5 + (lum - 0.5) * contrast, 0, 1)

    hue, sat, _ = colorsys.rgb_to_hsv(*[float(c) for c in tint])
    comp = hsv_hex(hue + 0.5, 0.75, 0.30)
    white = np.array([1.0, 0.985, 0.97], dtype=np.float32)
    black = np.array([0.010, 0.010, 0.014], dtype=np.float32)

    stops = np.array([0.0, 0.16, 0.38, 0.62, 0.82, 1.0], dtype=np.float32)
    cols = np.stack([
        mix(black, comp, 0.30),            # глубокие тени — холодный сумрак
        mix(black, comp * 0.7, 0.20),
        tint * 0.40,
        tint,                               # основной цвет
        mix(tint, white, 0.50),
        mix(tint, white, 0.90),            # блики почти белые
    ])
    out = np.stack([np.interp(lum, stops, cols[:, c]) for c in range(3)], axis=-1)

    # немного исходной хромы, чтобы картинка не стала плоской заливкой
    res = out * (1 - preserve) + img * preserve
    return np.clip(res, 0, 1)


# ----------------------------------------------------------------------------
# Полосы: срезы, растяжки, блюры
# ----------------------------------------------------------------------------

def box_blur_h(band: np.ndarray, radius: int) -> np.ndarray:
    """Горизонтальный box-blur через кумулятивные суммы — дешёвый «смаз»."""
    radius = max(1, int(radius))
    pad = np.pad(band, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=1, dtype=np.float64)
    cs = np.concatenate([np.zeros((band.shape[0], 1, 3)), cs], axis=1)
    width = 2 * radius + 1
    out = (cs[:, width:, :] - cs[:, :-width, :]) / width
    return out.astype(np.float32)


def motion_blur(band: np.ndarray, radius: int, dy: int = 0) -> np.ndarray:
    """Смаз вправо с одновременным сползанием по вертикали — след движения."""
    blur = box_blur_h(band, radius)
    if dy:
        blur = np.roll(blur, dy, axis=0)
        # подмешиваем исходник со сдвигом, чтобы получился «хвост»
        blur = np.clip(blur * 0.72 + np.roll(band, dy * 2, axis=0) * 0.28, 0, 1)
    return blur


def progressive_blur(band: np.ndarray, rng) -> np.ndarray:
    """Размытие нарастает слева направо (или наоборот) — как в референсах."""
    W = band.shape[1]
    radius = int(rng.integers(max(4, W // 45), max(6, W // 8)))
    blurred = box_blur_h(band, radius)
    blurred = box_blur_h(blurred, max(1, radius // 2))
    ramp = np.linspace(0.0, 1.0, W, dtype=np.float32) ** float(rng.uniform(0.7, 1.8))
    if rng.random() < 0.5:
        ramp = 1.0 - ramp
    w = ramp[None, :, None]
    return np.clip(band * (1 - w) + blurred * w, 0, 1)





def _pick_source(img: np.ndarray, rng, sw: int, sh: int, tries: int = 7) -> tuple[int, int]:
    """
    Выбирает кусок для растяжки с уклоном в светлые области: иначе на тёмном
    фоне полосы получаются почти чёрными и эффект теряется.
    """
    H, W, _ = img.shape
    cands, weights = [], []
    for _ in range(tries):
        sy = int(rng.integers(0, max(1, H - sh)))
        sx = int(rng.integers(0, max(1, W - sw)))
        cands.append((sy, sx))
        weights.append(float(luminance(img[sy:sy + sh, sx:sx + sw]).mean()) ** 2 + 0.01)
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for cand, w in zip(cands, weights):
        acc += w
        if r <= acc:
            return cand
    return cands[-1]

def _soften(patch: np.ndarray, rng, rx: float = 2.0, ry: float = 0.5) -> np.ndarray:
    """Лёгкое размытие растянутого куска: полоски перестают быть «цифровыми»."""
    im = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur((float(rng.uniform(0, rx)),
                                             float(rng.uniform(0, ry)))))
    return to_float(im)

def _blend_rect(out: np.ndarray, y: int, x0: int, patch: np.ndarray,
                alpha: float, rng, feather: int = 5) -> np.ndarray:
    """Вклеивает кусок с мягкими краями — он растворяется в кадре."""
    h, w = patch.shape[:2]
    if float(patch.mean()) < 0.12:  # тёмная заплатка читается как чёрный прямоугольник
        alpha *= 0.45
    a = np.full((h, w, 1), alpha, dtype=np.float32)

    f = _feather_width(min(feather, max(1, h // 3)))
    if f > 1:
        ramp = np.ones(h, dtype=np.float32)
        ramp[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
        ramp[-f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
        a = a * ramp[:, None, None]

    fx = _feather_width(min(max(4, w // 3), rng.integers(8, 30)))
    if fx > 1:
        ramp = np.ones(w, dtype=np.float32)
        ramp[:fx] = np.linspace(0.0, 1.0, fx, dtype=np.float32)
        ramp[-fx:] = np.linspace(1.0, 0.0, fx, dtype=np.float32)
        jitter = rng.uniform(0.8, 1.0, w).astype(np.float32)
        ramp = ramp * jitter
        a = a * ramp[None, :, None]

    region = out[y:y + h, x0:x0 + w]
    out[y:y + h, x0:x0 + w] = np.clip(region * (1 - a) + patch * a, 0, 1)
    return out


def _x_mask(width: int, x0: int, rng, feather: int = 14) -> np.ndarray:
    """Маска с растушёванными и рваными краями для частичной полосы."""
    W = max(width + x0, x0 + width)
    m = np.zeros(W, dtype=np.float32)
    x1 = min(W, x0 + width)
    m[x0:x1] = 1.0
    f = _feather_width(min(feather, max(1, (x1 - x0) // 2)))
    if f > 1:
        m[x0:x0 + f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
        m[x1 - f:x1] = np.linspace(1.0, 0.0, f, dtype=np.float32)
    if FEATHER <= 0:
        return m
    jitter = rng.uniform(0.75, 1.0, W).astype(np.float32)
    jitter = np.convolve(jitter, np.ones(9) / 9, mode="same").astype(np.float32)
    return np.clip(m * (0.7 + 0.3 * jitter), 0, 1)



def _channel_streak(band: np.ndarray, lengths: tuple[int, int, int]) -> np.ndarray:
    """
    Вытягивает яркие пиксели вбок отдельно по каналам. Разная длина у R, G и B
    даёт цветовую дисперсию вдоль штриха — то, что видно на референсах.
    """
    out = band.copy()
    for c, L in enumerate(lengths):
        ch = band[..., c:c + 1]
        acc = ch.copy()
        for k in range(2, max(3, L), 2):
            acc = np.maximum(acc, np.roll(ch, k, axis=1))
        out[..., c] = acc[..., 0]
    return out


def torn_streak(img: np.ndarray, rng, tint: np.ndarray) -> np.ndarray:
    """
    Рваный грязный штрих: кусок кадра вытянут вбок, каналы разъехались,
    внутри шум, края рваные. Ближе всего к помеченным зонам на референсе.
    """
    H, W, _ = img.shape
    h = int(rng.integers(max(18, H // 40), max(26, H // 12)))
    if h >= H:
        return img
    # выбираем полосу с ярким содержимым: из тёмной зоны штрихи не читаются
    y = int(rng.integers(0, H - h))
    best = -1.0
    for _ in range(6):
        cy = int(rng.integers(0, H - h))
        lum = float(luminance(img[cy:cy + h]).max())
        if lum > best:
            best, y = lum, cy
    width = max(60, int(W * float(rng.uniform(0.22, 0.66))))
    x0 = int(rng.integers(int(W * 0.02), max(int(W * 0.03), W - width - int(W * 0.02))))
    width = min(width, W - x0 - 4)
    if width < 40:
        return img

    band = img[y:y + h, x0:x0 + width]
    L = int(rng.integers(max(12, W // 34), max(18, W // 10)))
    direction = 1 if rng.random() < 0.65 else -1
    lengths = (int(L * rng.uniform(1.2, 1.7)), L, int(L * rng.uniform(0.6, 0.9)))

    patch = _channel_streak(band, lengths) if direction > 0 else _channel_streak(band[:, ::-1], lengths)[:, ::-1]

    patch = patch * 0.78 + band * 0.22                      # оставляем фактуру
    patch = np.clip((patch - 0.5) * float(rng.uniform(1.15, 1.5)) + 0.5, 0, 1)
    patch = np.clip(patch, 0, 1) ** float(rng.uniform(1.2, 2.4))
    # грязная вуаль: рвёт сплошную заливку на отдельные волокна
    dirt = rng.uniform(0.35, 1.15, (max(2, h // 5), max(2, width // 28))).astype(np.float32)
    dirt = resize_arr(np.repeat(dirt[..., None], 3, axis=2), width, h, Image.BICUBIC)[..., :1]
    patch = np.clip(patch * (0.55 + 0.6 * dirt), 0, 1)
    for rr in range(h):
        if rng.random() < 0.5:
            patch[rr] = np.roll(patch[rr], int(rng.integers(-24, 25)), axis=0)
    patch = np.clip(patch + rng.normal(0, 0.05, patch.shape).astype(np.float32), 0, 1)

    # рваная маска: растушёвка по краям плюс дыры по строкам
    m = _x_mask(width, 0, rng, feather=int(rng.integers(10, 40)))
    rows = rng.uniform(0.55, 1.0, (h, 1)).astype(np.float32)
    rows = np.convolve(rows[:, 0], np.ones(5) / 5, mode="same")[:, None].astype(np.float32)
    a = np.clip(m[None, :, None] * (0.65 + 0.35 * rows[..., None]), 0, 1)
    a = a * float(rng.uniform(0.92, 1.0))

    region = img[y:y + h, x0:x0 + width]
    out = img.copy()
    out[y:y + h, x0:x0 + width] = np.clip(region * (1 - a) + patch * a, 0, 1)

    # тонкая светлая кромка сверху — как у сбойной строки
    if rng.random() < 0.55:
        rim = mix(np.array([1.0, 1.0, 1.0], dtype=np.float32), tint, 0.35)
        ra = float(rng.uniform(0.15, 0.45))
        row = out[y:y + 1, x0:x0 + width]
        out[y:y + 1, x0:x0 + width] = np.clip(row * (1 - ra) + rim * ra, 0, 1)

    return out

def streak_bar(img: np.ndarray, rng, tint: np.ndarray) -> np.ndarray:
    """
    Короткий смазанный штрих внутри кадра: не на всю ширину, с вертикальными
    полосками, шумом, рваными краями и светлой кромкой снизу.
    """
    H, W, _ = img.shape
    h = int(rng.integers(max(14, H // 55), max(22, H // 14)))
    if h >= H:
        return img
    y = int(rng.integers(0, H - h))
    width = max(60, int(W * float(rng.uniform(0.24, 0.7))))
    x0 = int(rng.integers(int(W * 0.02), max(int(W * 0.03), W - width - int(W * 0.02))))
    width = min(width, W - x0 - int(W * 0.01))

    # источник — узкий кусок кадра рядом с полосой, чтобы цвет был «свой»
    sy = int(np.clip(y + rng.integers(-h, h), 0, H - 2))
    sh = int(rng.integers(1, max(2, min(4, H - sy))))
    sx = int(rng.integers(0, max(1, W - 8)))
    sw = int(rng.integers(4, 64))
    sw = min(sw, W - sx)
    src = img[sy:sy + sh, sx:sx + sw]
    if src.size == 0:
        return img

    patch = resize_arr(src, width, h)
    patch = patch[:, _warp_index(width, width, rng), :]      # вертикальные полоски
    patch = box_blur_h(patch, max(1, int(width / 45)))       # смаз вдоль штриха
    patch = np.clip((patch - 0.5) * float(rng.uniform(1.15, 1.6)) + 0.5, 0, 1)
    lum = patch @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    patch = np.clip(lum[..., None] + (patch - lum[..., None]) * float(rng.uniform(1.2, 1.8)), 0, 1)
    lum0 = patch.mean(axis=-1, keepdims=True)
    hi = np.clip((lum0 - 0.45) * 2.5, 0, 1)
    patch = np.clip(patch * 0.45 + patch * hi * 0.95, 0, 1)
    patch = np.clip(patch * rng.normal(1.0, 0.16, (1, width, 1)).astype(np.float32), 0, 1)
    # тёмные разрывы внутри
    for _ in range(int(rng.integers(1, 4))):
        ry = int(rng.integers(1, max(2, h - 1)))
        xa = int(rng.integers(0, max(1, width // 2)))
        xb = min(width, xa + int(rng.integers(width // 6, width)))
        patch[ry, xa:xb] = np.clip(patch[ry, xa:xb] * 0.25, 0, 1)

    # пара светлых строк внутри — как пробитый сигнал
    for _ in range(int(rng.integers(1, 4))):
        ry = int(rng.integers(1, max(2, h - 1)))
        patch[ry] = np.clip(patch[ry] + rng.uniform(0.15, 0.5), 0, 1)
    patch = np.clip(patch + rng.normal(0, 0.035, patch.shape).astype(np.float32), 0, 1)
    patch = _soften(patch, rng, 0.4, 0.5)

    out = img.copy()
    m = _x_mask(width, 0, rng, feather=int(rng.integers(8, 30)))
    m[:1] = 0.0
    m[-1:] = 0.0
    a = (m[None, :, None] * float(rng.uniform(0.92, 1.0)))
    region = out[y:y + h, x0:x0 + width]
    out[y:y + h, x0:x0 + width] = np.clip(region * (1 - a) + patch * a, 0, 1)

    # светлая кромка снизу и тень сверху
    rim = mix(np.array([1.0, 1.0, 1.0], dtype=np.float32), tint, float(rng.uniform(0.1, 0.5)))
    rim_a = float(rng.uniform(0.45, 0.9))
    edge = out[y + h - 1:y + h, x0:x0 + width]
    out[y + h - 1:y + h, x0:x0 + width] = np.clip(
        edge * (1 - rim_a) + (rim * 0.95 + 0.05) * rim_a, 0, 1)
    out[y:y + 1, x0:x0 + width] = out[y:y + 1, x0:x0 + width] * 0.55

    # рваные пропуски
    for _ in range(int(rng.integers(0, 4))):
        gx = int(rng.integers(0, max(1, width - 8)))
        gw = int(rng.integers(3, 22))
        gx1 = min(width, gx + gw)
        out[y:y + h, x0 + gx:x0 + gx1] = img[y:y + h, x0 + gx:x0 + gx1]

    return out


def stretch_to_fill(img: np.ndarray, rng, y: int, h: int, *, width_span: float | None = None,
                    alpha: float | None = None) -> np.ndarray:
    """
    Берёт небольшой кусок кадра и растягивает его на полосу.

    Именно это даёт внутри полосы градиентные линии разной толщины: узкий
    источник раздувается до ширины полосы, и каждая его колонка становится
    широкой мягкой полоской.
    """
    H, W, _ = img.shape
    sw = int(rng.integers(max(2, W // 220), max(4, W // 14)))
    sh = int(rng.integers(max(1, h // 12), max(2, h)))
    sx = int(rng.integers(0, max(1, W - sw)))
    sy = int(rng.integers(0, max(1, H - sh)))
    src = img[sy:sy + sh, sx:sx + sw]
    if src.size == 0:
        return img

    span = width_span if width_span is not None else float(rng.uniform(0.25, 0.78))
    target_w = max(24, int(W * span))
    x0 = int(rng.integers(0, max(1, W - target_w)))
    target_w = min(target_w, W - x0)
    patch = resize_arr(src, target_w, h)
    patch = np.clip((patch - 0.5) * float(rng.uniform(1.05, 1.3)) + 0.5, 0, 1)
    if rng.random() < 0.35:
        patch = _soften(patch, rng, 1.6, 0.5)

    a = alpha if alpha is not None else float(rng.uniform(0.45, 0.9))
    f = _feather_width(min(6, max(1, h // 3)))
    ramp = np.ones(h, dtype=np.float32)
    if f > 1:
        ramp[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
        ramp[-f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
    a_map = (ramp * a)[:, None, None]

    out = img.copy()
    region = out[y:y + h, x0:x0 + target_w]
    out[y:y + h, x0:x0 + target_w] = np.clip(region * (1 - a_map) + patch * a_map, 0, 1)
    return out


def _warp_index(n_out: int, n_src: int, rng) -> np.ndarray:
    """
    Нелинейная карта «выход → источник»: полосы внутри полосы получаются
    разной толщины, а не равномерной сеткой.
    """
    n_src = max(1, n_src)
    k = int(rng.integers(3, 11))
    xs = np.sort(rng.uniform(0.0, 1.0, k))
    xs = np.concatenate([[0.0], xs, [1.0]])
    ys = rng.uniform(0, n_src - 1, k + 2)
    idx = np.interp(np.linspace(0.0, 1.0, n_out), xs, ys)
    return np.clip(np.round(idx).astype(int), 0, n_src - 1)


def gradient_band(img: np.ndarray, rng, y: int, h: int) -> np.ndarray:
    """Мягкое цветовое поле внутри полосы — растянутый и размытый кусок кадра."""
    H, W, _ = img.shape
    sw = int(rng.integers(4, 44))
    sh = int(rng.integers(3, max(4, min(h, H))))
    sy, sx = _pick_source(img, rng, sw, sh)
    src = img[sy:sy + sh, sx:sx + sw]
    target_w = max(24, int(W * float(rng.uniform(0.25, 0.8))))
    tx = int(rng.integers(0, max(1, W - target_w)))
    target_w = min(target_w, W - tx)
    patch = resize_arr(src, target_w, h, Image.BICUBIC)
    im = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
    patch = to_float(im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 4.0)))))
    a = float(rng.uniform(0.72, 1.0))
    out = img.copy()
    return _blend_rect(out, y, tx, patch, a, rng, feather=8)

def stretch_vertical_lines(img: np.ndarray, rng, y: int, h: int) -> np.ndarray:
    """Узкий кусок тянется по ширине — вертикальные полоски разной толщины."""
    H, W, _ = img.shape
    x0 = int(rng.integers(0, max(1, W - 8)))
    cw = int(rng.integers(max(2, W // 300), max(5, W // 18)))
    cw = min(cw, W - x0)
    if cw < 2:
        return img
    src = img[y:y + h, x0:x0 + cw]
    target_w = max(24, int(W * float(rng.uniform(0.28, 0.8))))
    tx = int(rng.integers(0, max(1, W - target_w)))
    target_w = min(target_w, W - tx)
    idx = _warp_index(target_w, cw, rng)
    patch = src[:, idx, :]
    if rng.random() < 0.6:
        patch = _soften(patch, rng, 2.0, 0.4)
    a = float(rng.uniform(0.7, 1.0))
    out = img.copy()
    return _blend_rect(out, y, tx, patch, a, rng)


def stretch_horizontal_lines(img: np.ndarray, rng, y: int, h: int) -> np.ndarray:
    """Тонкая строка растягивается по высоте — горизонтальные линии разной толщины."""
    H, W, _ = img.shape
    sy = int(rng.integers(0, max(1, H - 3)))
    sh = int(rng.integers(1, max(2, min(7, H - sy))))
    target_w = max(24, int(W * float(rng.uniform(0.28, 0.8))))
    tx = int(rng.integers(0, max(1, W - target_w)))
    target_w = min(target_w, W - tx)
    src = img[sy:sy + sh, tx:tx + target_w]
    idx = _warp_index(h, sh, rng)
    patch = src[idx, :, :]
    if rng.random() < 0.6:
        patch = _soften(patch, rng, 0.4, 1.6)
    a = float(rng.uniform(0.65, 1.0))
    out = img.copy()
    return _blend_rect(out, y, tx, patch, a, rng)


def pixel_sort_band(band: np.ndarray, rng) -> np.ndarray:
    """Классический pixel sort по строке: яркие пиксели уезжают вправо."""
    out = band.copy()
    h, w, _ = band.shape
    rows = rng.choice(h, size=min(h, int(rng.integers(3, 12))), replace=False)
    lum = luminance(band)
    for r in rows:
        row_l = lum[r]
        thr = np.quantile(row_l, rng.uniform(0.55, 0.75))
        idx = np.flatnonzero(row_l > thr)
        if idx.size < 8:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        for seg in np.split(idx, splits + 1):
            if seg.size < 8:
                continue
            order = seg[np.argsort(row_l[seg])[::-1]]
            out[r, seg] = band[r, order]
    return out


def slice_glitch(img: np.ndarray, rng, tint: np.ndarray, amount: float = 1.0,
                 dirt: float = 1.0) -> np.ndarray:
    H, W, _ = img.shape
    out = img.copy()
    count = int(round(rng.integers(9, 15) * amount))

    def partial(band: np.ndarray, y: int, h: int, a_scale: float = 1.0) -> None:
        """Вклеивает результат только на часть ширины, с рваными краями."""
        span = float(rng.uniform(0.25, 0.8))
        width = max(24, int(W * span))
        x0 = int(rng.integers(0, max(1, W - width)))
        width = min(width, W - x0)
        m = _x_mask(width, 0, rng, feather=int(rng.integers(6, 30)))
        a = (m[None, :, None] * float(rng.uniform(0.55, 1.0) * a_scale))
        region = out[y:y + h, x0:x0 + width]
        out[y:y + h, x0:x0 + width] = np.clip(
            region * (1 - a) + band[:, x0:x0 + width] * a, 0, 1)

    prev_y2, prev_dx = None, None
    for _ in range(count):
        y = int(rng.integers(0, H - 2))
        h = int(rng.integers(3, max(4, int(H * 0.12))))
        y2 = min(H, y + h)
        h = y2 - y
        if h < 2:
            continue

        roll = rng.random()

        if roll < 0.05:  # рваный штрих с дисперсией
            out = torn_streak(out, rng, tint)

        elif roll < 0.09:  # короткий смазанный штрих
            out = streak_bar(out, rng, tint)

        elif roll < 0.16:  # мягкое градиентное поле
            out = gradient_band(out, rng, y, h)

        elif roll < 0.34:  # растяжка куска внутри полосы
            if rng.random() < 0.28:  # иногда полоса крупнее — большой смаз
                h = int(rng.integers(max(4, int(H * 0.10)), max(6, int(H * 0.20))))
                y = int(rng.integers(0, max(1, H - h)))
                y2 = min(H, y + h)
                h = y2 - y
            out = stretch_to_fill(out, rng, y, h, alpha=float(rng.uniform(0.7, 0.95)))

        elif roll < 0.46:  # вертикальные полоски внутри полосы
            out = stretch_vertical_lines(out, rng, y, h)

        elif roll < 0.54:  # горизонтальные градиентные линии
            out = stretch_horizontal_lines(out, rng, y, h)

        elif roll < 0.68:  # срез со сдвигом, часто «пачкой» соседних полос
            grouped = (prev_dx is not None and prev_y2 is not None
                       and prev_y2 < H - 4 and rng.random() < 0.45)
            if grouped:
                y = prev_y2
                h = int(rng.integers(2, max(3, int(H * 0.05))))
                y2 = min(H, y + h)
                h = y2 - y
                dx = prev_dx
            else:
                dx = int(rng.normal(0, W * 0.10))
            if h < 1 or abs(dx) < 4:
                continue
            roll_band = np.roll(out[y:y2], dx, axis=1)
            if rng.random() < 0.55:
                partial(roll_band, y, h, a_scale=0.9)
            else:
                out[y:y2] = roll_band
            prev_y2, prev_dx = y2, dx
            if rng.random() < 0.5:  # светлая кромка на стыке
                out[y2 - 1:y2] = np.clip(out[y2 - 1:y2] * 0.35 + 0.6, 0, 1)
            continue

        elif roll < 0.80:  # progressive blur
            band = out[y:y2]
            partial(progressive_blur(band, rng), y, h, a_scale=0.9)

        elif roll < 0.90:  # motion blur со сползанием
            radius = int(rng.integers(max(6, W // 40), max(10, W // 6)))
            dy = int(rng.integers(-3, 4))
            blurred = np.clip(out[y:y2] * 0.25 + motion_blur(out[y:y2], radius, dy) * 0.75, 0, 1)
            partial(blurred, y, h, a_scale=0.85)

        elif roll < 0.95:  # призрак: копия полосы со сдвигом
            dx = int(rng.integers(-W // 3, W // 3))
            band = out[y:y2]
            ghost = np.roll(band, dx, axis=1)
            a = float(rng.uniform(0.2, 0.5))
            partial(np.clip(band * (1 - a) + ghost * a, 0, 1), y, h, a_scale=0.9)

        elif roll < 0.98:  # грязная шумная полоса
            band = out[y:y2]
            noise = rng.normal(0, 0.05 + 0.05 * dirt, (h, W, 3)).astype(np.float32)
            band = np.clip(band + noise, 0, 1)
            for _ in range(int(rng.integers(2, 8))):
                yy = int(rng.integers(0, h))
                xa = int(rng.integers(0, W // 2))
                xb = int(min(W, xa + rng.integers(W // 12, W)))
                if rng.random() < 0.5:
                    band[yy, xa:xb] = np.clip(band[yy, xa:xb] * 0.35, 0, 1)
                else:
                    band[yy, xa:xb] = np.clip(band[yy, xa:xb] + 0.35, 0, 1)
            partial(band, y, h, a_scale=0.9)

        else:  # pixel sort
            out[y:y2] = pixel_sort_band(out[y:y2], rng)

        prev_y2, prev_dx = None, None

    # прямоугольные «заплатки» из растянутого соседнего куска
    for _ in range(int(rng.integers(1, 4) * amount)):
        bh = int(rng.integers(max(2, H // 50), max(4, H // 10)))
        y = int(rng.integers(0, max(1, H - bh)))
        span = float(rng.uniform(0.06, 0.30))
        out = stretch_to_fill(out, rng, y, bh, width_span=span)

    # добиваем короткими штрихами и рваными полосами
    for _ in range(int(rng.integers(2, 5) * amount)):
        out = streak_bar(out, rng, tint)
    for _ in range(int(rng.integers(2, 4) * amount)):
        out = torn_streak(out, rng, tint)

    return out


# ----------------------------------------------------------------------------
# Грязь: линии, царапины, пыль
# ----------------------------------------------------------------------------

def dirty_lines(img: np.ndarray, rng, tint: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Тонкие линии с зерном и рваными краями — «грязный сигнал»."""
    H, W, _ = img.shape
    out = img.copy()
    for _ in range(int(rng.integers(2, 6) * amount)):
        h = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
        y = int(rng.integers(0, H - h))
        x0 = int(rng.integers(-W // 6, W // 2))
        x1 = int(min(W, x0 + rng.integers(W // 3, int(W * 1.2))))
        if x1 <= 0:
            continue
        xa = max(0, x0)

        line = out[y:y + h, xa:x1]
        if line.size == 0:
            continue

        kind = rng.random()
        if kind < 0.45:  # светлая грязная линия
            base = mix(np.array([1.0, 1.0, 1.0], dtype=np.float32), tint,
                       float(rng.uniform(0.1, 0.6)))
            a = float(rng.uniform(0.10, 0.45))
            noise = rng.normal(0, 0.10, line.shape).astype(np.float32)
            patch = np.clip(base[None, None, :] + noise, 0, 1)
            out[y:y + h, xa:x1] = np.clip(line * (1 - a) + patch * a, 0, 1)
        elif kind < 0.75:  # тёмная линия
            a = float(rng.uniform(0.15, 0.55))
            noise = rng.normal(0, 0.06, line.shape).astype(np.float32)
            out[y:y + h, xa:x1] = np.clip(line * (1 - a) + noise * 0.5 + 0.01, 0, 1)
        else:  # рваная: часть строки светлая, часть тёмная
            mid = xa + (x1 - xa) // 2
            out[y:y + h, xa:mid] = np.clip(out[y:y + h, xa:mid] + rng.uniform(0.1, 0.4), 0, 1)
            out[y:y + h, mid:x1] = np.clip(out[y:y + h, mid:x1] * rng.uniform(0.4, 0.8), 0, 1)

    # вертикальные обрывы записи
    for _ in range(int(rng.integers(0, 3) * amount)):
        w = int(rng.integers(1, 7))
        x = int(rng.integers(0, max(1, W - w)))
        y0 = int(rng.integers(0, H // 2))
        hlen = int(rng.integers(H // 6, H))
        y1 = min(H, y0 + hlen)
        strip = out[y0:y1, x:x + w]
        if strip.size == 0:
            continue
        noise = rng.normal(0, 0.08, strip.shape).astype(np.float32)
        a = float(rng.uniform(0.15, 0.5))
        out[y0:y1, x:x + w] = np.clip(strip * (1 - a) + (strip + noise) * a, 0, 1)

    return out


def grunge(img: np.ndarray, rng, amount: float = 1.0) -> np.ndarray:
    """Пыль, пятна и низкочастотная грязь — фон перестаёт быть чистым."""
    H, W, _ = img.shape
    out = img.copy()

    # низкочастотная грязная вуаль
    small = rng.normal(1.0, 0.06, (max(2, H // 40), max(2, W // 40), 1)).astype(np.float32)
    veil = resize_arr(np.repeat(small, 3, axis=2), W, H, Image.BICUBIC)
    out = np.clip(out * (0.92 + 0.16 * veil), 0, 1)

    # пятна
    for _ in range(int(rng.integers(4, 12) * amount)):
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        r = int(rng.integers(6, max(10, H // 8)))
        yy0, yy1 = max(0, cy - r), min(H, cy + r)
        xx0, xx1 = max(0, cx - r), min(W, cx + r)
        if yy1 <= yy0 or xx1 <= xx0:
            continue
        yy, xx = np.mgrid[yy0:yy1, xx0:xx1]
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(1, r)
        mask = np.clip(1 - d, 0, 1)[..., None] ** 2
        out[yy0:yy1, xx0:xx1] *= (1 - 0.35 * mask)

    # пылинки и царапины
    for _ in range(int(rng.integers(20, 60) * amount)):
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, W))
        sx = int(rng.integers(1, 4))
        sy = int(rng.integers(1, 3))
        bright = rng.random() < 0.6
        val = rng.uniform(0.4, 1.0) if bright else rng.uniform(0.0, 0.2)
        out[y:y + sy, x:x + sx] = np.clip(out[y:y + sy, x:x + sx] * 0.3 + val * 0.7, 0, 1)

    return out


def jpeg_dirt(img: np.ndarray, rng, strength: float = 0.0) -> np.ndarray:
    """Прогон через JPEG с низким качеством — настоящая блочная грязь кодека."""
    if strength <= 0:
        return img
    quality = int(np.clip(86 - 34 * strength + rng.normal(0, 3), 40, 92))
    buff = io.BytesIO()
    to_pil(img).save(buff, format="JPEG", quality=quality, subsampling=2)
    buff.seek(0)
    return to_float(Image.open(buff).convert("RGB"))


# ----------------------------------------------------------------------------
# Полосы-оверлеи
# ----------------------------------------------------------------------------

def overlay_bars(img: np.ndarray, rng, tint: np.ndarray, amount: float = 1.0) -> np.ndarray:
    H, W, _ = img.shape
    color_map = np.zeros_like(img)
    alpha_map = np.zeros((H, W), dtype=np.float32)

    for _ in range(int(rng.integers(2, 5) * amount)):
        h = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 6, 10]))
        y = int(rng.integers(0, H))
        if rng.random() < 0.1:  # редкая линия во всю ширину
            x0c, x1 = 0, W
        else:
            width = max(80, int(W * rng.uniform(0.15, 0.75)))
            x0c = int(rng.integers(0, max(1, W - width)))
            x1 = min(W, x0c + width)

        dark = rng.random() < 0.22
        if dark:
            colour = np.array([0.01, 0.01, 0.015], dtype=np.float32)
            alpha = float(rng.uniform(0.14, 0.42))
        else:
            colour = mix(np.array([1.0, 1.0, 1.0], dtype=np.float32), tint,
                         float(rng.uniform(0.10, 0.6)))
            alpha = float(rng.uniform(0.10, 0.46))

        if rng.random() < 0.15:  # редкая широкая «пластина»
            h = int(rng.integers(int(H * 0.02), int(H * 0.07)))
            alpha *= 0.18

        color_map[y:y + h, x0c:x1] = colour
        alpha_map[y:y + h, x0c:x1] = alpha

    soft = np.asarray(
        Image.fromarray((alpha_map * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    ).astype(np.float32) / 255.0
    alpha_map = np.clip(soft, 0, 1)[..., None]

    screen = 1 - (1 - img) * (1 - color_map)
    out = img * (1 - alpha_map) + screen * alpha_map
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------
# Текст
# ----------------------------------------------------------------------------

FONT_CANDIDATES = [
    os.path.expanduser("~/Library/Fonts/MedeoSans[opsz,wght].ttf"),
    os.path.expanduser("~/Library/Fonts/MedeoMono[wght].ttf"),
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def pick_font(path: str | None, size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, size)
    order = list(FONT_CANDIDATES)
    if mono:
        order = [os.path.expanduser("~/Library/Fonts/MedeoMono[wght].ttf")] + order
    for candidate in order:
        if candidate and os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def random_line(rng: random.Random, min_len: int = 2, max_len: int = 6) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(CHARSET) for _ in range(n))


def draw_tracked(draw: ImageDraw.ImageDraw, x: float, y: float, text: str,
                 font, tracking: float, fill) -> float:
    """Рисует строку с разрядкой и возвращает её ширину."""
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking
    return x


def text_layer(size: tuple[int, int], rng: random.Random, tint: np.ndarray,
               font_path: str | None, mono: bool, density: float = 1.0) -> Image.Image:
    W, H = size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    clusters = max(3, int(rng.randint(5, 9) * density))
    for _ in range(clusters):
        if rng.random() < 0.18:
            fs = int(H * rng.uniform(0.024, 0.032))   # редкий крупный кластер
        else:
            fs = int(H * rng.uniform(0.014, 0.022))
        font = pick_font(font_path, fs, mono)
        tracking = fs * rng.uniform(0.14, 0.9)

        zone = rng.random()
        if zone < 0.3:
            x = rng.uniform(0.03, 0.22) * W
        elif zone < 0.55:
            x = rng.uniform(0.30, 0.62) * W
        elif zone < 0.8:
            x = rng.uniform(0.62, 0.88) * W
        else:
            x = rng.uniform(0.86, 0.99) * W
        y = rng.uniform(0.02, 0.9) * H

        lines = rng.randint(1, 6)
        lh = fs * rng.uniform(1.15, 1.55)
        vertical = rng.random() < 0.18
        accent = rng.random() < 0.3
        if accent:
            fill = tuple(int(c * 255) for c in mix(np.array([1, 1, 1]), tint, 0.45)) + \
                   (rng.randint(150, 255),)
        else:
            fill = (250, 250, 252, rng.randint(140, 250))

        if vertical:
            step = fs * rng.uniform(1.15, 1.9)
            for li in range(rng.randint(3, 9)):
                draw.text((x, y + li * step), rng.choice(CHARSET), font=font, fill=fill)
            continue

        for li in range(lines):
            if rng.random() < 0.18:
                a = random_line(rng, 1, 3)
                b = random_line(rng, 1, 3)
                w = draw_tracked(draw, x, y + li * lh, a, font, tracking, fill)
                gap = fs * rng.uniform(2.0, 5.5)
                draw_tracked(draw, w + gap, y + li * lh, b, font, tracking, fill)
            else:
                text = random_line(rng, 1, 5)
                off = rng.choice([0, 0, 0, fs * rng.uniform(1.0, 3.0)])
                draw_tracked(draw, x + off, y + li * lh, text, font, tracking, fill)

    # тонкие колонки по краям — фирменная деталь референсов
    for _ in range(int(rng.randint(3, 6) * density)):
        fs = int(H * rng.uniform(0.012, 0.018))
        font = pick_font(font_path, fs, mono)
        right = rng.random() < 0.5
        x = rng.uniform(0.955, 0.995) * W if right else rng.uniform(0.005, 0.035) * W
        y = rng.uniform(0.0, 0.55) * H
        step = fs * rng.uniform(1.1, 2.0)
        for i in range(rng.randint(4, 14)):
            ch = rng.choice(CHARSET) if rng.random() < 0.7 else ":"
            draw.text((x, y + i * step), ch, font=font,
                      fill=(235, 240, 255, rng.randint(70, 190)))

    return layer


# ----------------------------------------------------------------------------
# Финиш
# ----------------------------------------------------------------------------

def finish(img: np.ndarray, rng, grain: float = 0.03, vignette: float = 0.32,
           aberration: float = 1.0, blur: float = 0.6) -> np.ndarray:
    H, W, _ = img.shape
    out = img.copy()

    if aberration > 0:
        d = int(round(rng.uniform(0.5, 2.5) * aberration))
        if d:
            out[..., 0] = np.roll(out[..., 0], d, axis=1)
            out[..., 2] = np.roll(out[..., 2], -d, axis=1)

    # блум: светлые зоны слегка «светятся»
    bright = np.clip(out - 0.55, 0, 1) * 2.0
    if bright.max() > 0.02:
        im = Image.fromarray((np.clip(bright, 0, 1) * 255).astype(np.uint8))
        im = im.filter(ImageFilter.GaussianBlur(max(2, int(min(H, W) * 0.015))))
        halo = to_float(im) * 0.30
        out = 1 - (1 - out) * (1 - halo)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2, H / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / math.sqrt(2)
    out = out * (1 - vignette * (r ** 2.2))[..., None]

    lum = luminance(out)[..., None]
    noise = rng.normal(0, 1, (H, W, 1)).astype(np.float32)
    amp = grain * (0.55 + 0.9 * (1 - np.abs(lum - 0.5) * 2))
    out = out + noise * amp + rng.normal(0, 1, (H, W, 3)).astype(np.float32) * 0.008

    if blur > 0:
        im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
        out = to_float(im.filter(ImageFilter.GaussianBlur(blur)))

    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------
# Рендер одного варианта
# ----------------------------------------------------------------------------

def render(src: np.ndarray, seed: int, opts) -> np.ndarray:
    global FEATHER
    FEATHER = float(getattr(opts, "feather", 1.0))
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)

    no_tint = str(opts.tint).strip().lower() in ("none", "no", "original", "off")
    tint = np.array([1.0, 1.0, 1.0], dtype=np.float32) if no_tint \
        else resolve_tint(opts.tint, src, rng)
    out = neutral_grade(src, contrast=opts.contrast) if no_tint \
        else duotone(src, tint, preserve=opts.preserve, contrast=opts.contrast, dirt=opts.dirt)
    out = progressive_areas(out, np.random.default_rng(seed + 4242), opts.bg_blur)
    out = slice_glitch(out, rng, tint, amount=opts.intensity, dirt=opts.dirt)
    out = jpeg_dirt(out, rng, opts.jpeg_dirt)
    out = dirty_lines(out, rng, tint, amount=opts.dirt)
    out = grunge(out, rng, amount=opts.dirt)

    size = (src.shape[1], src.shape[0])

    if opts.text:
        t1 = text_layer(size, prng, tint, opts.font, opts.mono, density=opts.text_density)
        base = to_pil(out)
        base.paste(t1, (0, 0), t1)
        out = to_float(base)

    out = overlay_bars(out, rng, tint, amount=opts.bars)

    if opts.text:
        t2 = text_layer(size, prng, tint, opts.font, opts.mono,
                        density=opts.text_density * 0.7)
        base = to_pil(out)
        base.paste(t2, (0, 0), t2)
        out = to_float(base)

    out = finish(out, rng, grain=opts.grain * (0.7 + 0.6 * opts.dirt),
                 vignette=opts.vignette, aberration=opts.aberration, blur=opts.blur)
    return out


# ----------------------------------------------------------------------------
# Контактный лист
# ----------------------------------------------------------------------------

def make_sheet(paths: list[str], out_path: str, cols: int = 2, width: int = 880) -> None:
    """Контактный лист из готовых вариантов — удобно выбирать глазами."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    if not ims:
        return
    ratio = ims[0].height / ims[0].width
    tw, th = width, int(width * ratio)
    rows = (len(ims) + cols - 1) // cols
    gap = 10
    sheet = Image.new("RGB", (cols * tw + gap * (cols + 1), rows * th + gap * (rows + 1)),
                      (10, 10, 12))
    font = pick_font(None, max(14, tw // 46))
    draw = ImageDraw.Draw(sheet)
    for i, (im, path) in enumerate(zip(ims, paths)):
        x = gap + (i % cols) * (tw + gap)
        y = gap + (i // cols) * (th + gap)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        draw.text((x + 10, y + th - font.size - 12), os.path.basename(path), font=font,
                  fill=(190, 190, 200))
    sheet.save(out_path, quality=90)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Рандомизированный глитч для фотографий")
    p.add_argument("input", help="исходное изображение или папка")
    p.add_argument("-o", "--out", default="glitch-out", help="папка для результатов")
    p.add_argument("-n", "--count", type=int, default=4, help="сколько вариантов")
    p.add_argument("--seed", type=int, default=None, help="базовый seed (по умолчанию случайный)")
    p.add_argument("--tint", default="none",
                   help="auto | none | blue | orange | green | magenta | amber | cyan | violet | #hex")
    p.add_argument("--width", type=int, default=None, help="ширина результата")
    p.add_argument("--height", type=int, default=None, help="высота результата")
    p.add_argument("--intensity", type=float, default=1.0, help="сила полос, 0..2")
    p.add_argument("--dirt", type=float, default=1.2, help="грязь: шум, линии, пятна, 0..2")
    p.add_argument("--bg-blur", type=float, default=1.0,
                   help="прогрессивное размытие участков: 0 — выкл, 1 — по умолчанию, 2 — сильное")
    p.add_argument("--bars", type=float, default=1.0, help="количество полос-оверлеев, 0..2")
    p.add_argument("--jpeg-dirt", type=float, default=0.45, help="блочность кодека, 0..1")
    p.add_argument("--preserve", type=float, default=0.12, help="доля исходного цвета, 0..1")
    p.add_argument("--contrast", type=float, default=1.08, help="контраст тонировки")
    p.add_argument("--grain", type=float, default=0.03, help="зерно")
    p.add_argument("--vignette", type=float, default=0.32, help="виньетка")
    p.add_argument("--blur", type=float, default=0.6, help="мягкость")
    p.add_argument("--aberration", type=float, default=1.0, help="хроматическая аберрация")
    p.add_argument("--feather", type=float, default=1.0,
                   help="мягкость краёв полос: 0 — рубленые, 1 — обычные, 2 — мягкие")
    p.add_argument("--text", choices=["on", "off"], default="on")
    p.add_argument("--text-density", type=float, default=1.0, help="плотность глифов")
    p.add_argument("--font", default=None, help="путь к TTF/OTF для глифов")
    p.add_argument("--mono", action="store_true", help="моноширинный шрифт для глифов")
    p.add_argument("--format", default="png", choices=["png", "jpg", "webp"])
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--sheet", action="store_true", help="собрать контактный лист")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    args.text = args.text == "on"

    if not os.path.exists(args.input):
        print(f"нет файла: {args.input}", file=sys.stderr)
        return 1

    if os.path.isdir(args.input):
        exts = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
        sources = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                         if f.lower().endswith(exts))
    else:
        sources = [args.input]

    os.makedirs(args.out, exist_ok=True)
    base_seed = args.seed if args.seed is not None else random.SystemRandom().randint(1, 10 ** 6)
    print(f"seed: {base_seed}, файлов: {len(sources)}")

    made: list[str] = []
    for source in sources:
        src = load_image(source, args.width, args.height)
        stem = os.path.splitext(os.path.basename(source))[0]
        for i in range(args.count):
            seed = base_seed + i * 977
            out = render(src, seed, args)
            name = f"{stem}-glitch-{base_seed}-{i:02d}.{args.format}"
            path = os.path.join(args.out, name)
            if args.format == "jpg":
                to_pil(out).save(path, quality=args.quality, subsampling=1)
            elif args.format == "webp":
                to_pil(out).save(path, quality=args.quality, method=6)
            else:
                to_pil(out).save(path)
            made.append(path)
            print(f"  {path}  (seed {seed})")

    if args.sheet and made:
        sheet_path = os.path.join(args.out, f"sheet-{base_seed}.jpg")
        make_sheet(made, sheet_path)
        print(f"  лист: {sheet_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
