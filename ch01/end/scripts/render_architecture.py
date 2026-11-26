"""Render the chapter 1 pipeline architecture for a given section.

Example:
    uv run python scripts/render_architecture.py --stage 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "ch01"

# name, real detail, placeholder detail, first stage where it is real
STEPS = [
    ("文字", "the cat sat", "the cat sat", 0),
    ("编号", "CharacterTokenizer（分词器）", "占位：ord(ch) % 68", 2),
    ("向量", "Embedding（嵌入表）", "占位：累加取余", 3),
    ("位置", "位置嵌入", "占位：不加", 4),
    ("混合", "因果多头注意力", "占位：累加取余", 6),
    ("归一化/筛选", "LayerNorm + GELU", "占位：恒等", 5),
    ("分数", "Linear 输出", "占位：取余分数", 3),
    ("下一个字符", "argmax", "argmax", 0),
]


def _box(index: int, name: str, detail: str, real: bool) -> str:
    top = 80 + index * 92
    fill = "#e0ecff" if real else "#f8fafc"
    stroke = "#3b6ea5" if real else "#94a3b8"
    dash = "" if real else ' stroke-dasharray="6 4"'
    status = "真" if real else "占位"
    status_color = "#047857" if real else "#64748b"
    return (
        f'<rect x="110" y="{top}" width="540" height="64" rx="10" '
        f'fill="{fill}" stroke="{stroke}"{dash}/>\n'
        f'<text x="135" y="{top + 26}" font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" '
        f'font-size="16" font-weight="600" fill="#1f2937">{name}</text>\n'
        f'<text x="135" y="{top + 50}" font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" '
        f'font-size="13" fill="#475569">{detail}</text>\n'
        f'<text x="625" y="{top + 39}" text-anchor="end" '
        f'font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" font-size="13" '
        f'fill="{status_color}">{status}</text>'
    )


def render(stage: int) -> str:
    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 880" '
            'role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">第 {stage} 节结束时的管线架构</title>',
        '<desc id="desc">实线表示本节已经实现的组件，虚线表示仍在使用的占位组件。</desc>',
        '<rect width="760" height="880" fill="#ffffff"/>',
        (
            f'<text x="30" y="36" font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" '
            f'font-size="18" font-weight="600" fill="#1f2937">第 {stage} 节结束时的管线</text>'
        ),
        (
            '<text x="730" y="36" text-anchor="end" '
            'font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" font-size="13" '
            'fill="#64748b">实线 = 已实现，虚线 = 占位</text>'
        ),
    ]

    for index, (name, real_detail, placeholder_detail, required) in enumerate(STEPS):
        real = required == 0 or stage >= required
        detail = real_detail if real else placeholder_detail
        if name == "分数" and real and stage >= 7:
            detail = "TinyGPT 输出"
        parts.append(_box(index, name, detail, real))
        if index + 1 < len(STEPS):
            top = 80 + index * 92
            parts.append(
                f'<line x1="380" y1="{top + 68}" x2="380" y2="{top + 88}" '
                'stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    parts.append(
        '<text x="30" y="840" font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" '
        'font-size="13" fill="#475569">占位组件只为了让管线先跑起来；后面的小节逐个替换它们。</text>'
    )
    parts.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or OUTPUT_DIR / f"architecture-{args.stage}.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(args.stage), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
