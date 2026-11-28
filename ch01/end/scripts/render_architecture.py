"""Render the chapter 1 pipeline architecture for a given section.

Only the steps that exist at that section are drawn; later components are not
shown as future placeholders.

Example:
    uv run python scripts/render_architecture.py --stage 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "assets" / "ch01"

# name, real detail, placeholder detail, real stage, visible(stage)
# Sections 1.5-1.6 show LayerNorm/GELU and attention as separate stages; from
# 1.7 they are folded into TransformerBlock, so the aggregate replaces them.
STEPS = [
    ("文字", "the cat sat", "the cat sat", 0, lambda stage: True),
    (
        "编号",
        "CharacterTokenizer（分词器）",
        "占位：ord(ch) % 68",
        2,
        lambda stage: True,
    ),
    ("数字", "Embedding（嵌入表）", "", 3, lambda stage: stage >= 3),
    ("位置", "位置嵌入", "", 4, lambda stage: stage >= 4),
    ("调匀/筛选", "LayerNorm + GELU", "", 5, lambda stage: 5 <= stage <= 6),
    ("混合", "因果多头注意力", "", 6, lambda stage: stage == 6),
    (
        "TransformerBlock ×2",
        "两个 block 串联：注意力 + MLP，各带残差",
        "",
        7,
        lambda stage: stage >= 7,
    ),
    ("最终 LayerNorm", "LayerNorm", "", 7, lambda stage: stage >= 7),
    ("共享输出", "Z Eᵀ → logits", "", 7, lambda stage: stage >= 7),
    (
        "答案",
        "68 个分数 → argmax → 字符",
        "占位：编号 → 字符",
        7,
        lambda stage: True,
    ),
]


def _answer_detail(stage: int) -> str:
    if stage >= 7:
        return "68 个分数 → argmax → 字符"
    if stage >= 3:
        return "临时：输出向量 → 分数 → 字符"
    return "占位：编号 → 字符"


def _box(top: int, name: str, detail: str, real: bool) -> str:
    fill = "#e0ecff" if real else "#f8fafc"
    stroke = "#3b6ea5" if real else "#94a3b8"
    dash = "" if real else ' stroke-dasharray="6 4"'
    status = "真" if real else "占位"
    status_color = "#047857" if real else "#64748b"
    return (
        f'<rect x="110" y="{top}" width="540" height="64" rx="10" '
        f'fill="{fill}" stroke="{stroke}"{dash}/>\n'
        f'<text x="135" y="{top + 26}" '
        f"font-family=\"system-ui, 'Noto Sans CJK SC', sans-serif\" "
        f'font-size="16" font-weight="600" fill="#1f2937">{name}</text>\n'
        f'<text x="135" y="{top + 50}" '
        f"font-family=\"system-ui, 'Noto Sans CJK SC', sans-serif\" "
        f'font-size="13" fill="#475569">{detail}</text>\n'
        f'<text x="625" y="{top + 39}" text-anchor="end" '
        f"font-family=\"system-ui, 'Noto Sans CJK SC', sans-serif\" "
        f'font-size="13" fill="{status_color}">{status}</text>'
    )


def render(stage: int) -> str:
    steps = [step for step in STEPS if step[4](stage)]
    has_placeholder = any(stage < step[3] for step in steps)
    chapter_complete = not has_placeholder and stage >= 8
    if has_placeholder:
        desc = "实线表示本节已经实现的步骤，虚线表示仍在使用的占位。"
        legend = "实线 = 已实现，虚线 = 占位"
        footer = "只画本节已有的步骤；后面的组件在对应小节才出现。"
        desc += "后续小节的组件只在对应小节出现。"
    elif chapter_complete:
        desc = "实线表示已经实现的步骤；没有占位。"
        legend = "实线 = 已实现"
        footer = "只画本章已有的步骤；没有占位。"
    else:
        desc = "实线表示本节已经实现的步骤；本节没有占位。"
        legend = "实线 = 已实现"
        footer = "只画本节已有的步骤；本节没有占位。"
        desc += "后续小节的组件只在对应小节出现。"
    title = "第 1 章完成后的管线" if chapter_complete else f"第 {stage} 节结束时的管线"
    height = 80 + len(steps) * 92 + 60
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 {height}" '
            'role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{title}架构</title>',
        f'<desc id="desc">{desc}</desc>',
        f'<rect width="760" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="30" y="36" font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" '
            f'font-size="18" font-weight="600" fill="#1f2937">{title}</text>'
        ),
        (
            '<text x="730" y="36" text-anchor="end" '
            'font-family="system-ui, \'Noto Sans CJK SC\', sans-serif" font-size="13" '
            f'fill="#64748b">{legend}</text>'
        ),
    ]

    for index, (
        name,
        real_detail,
        placeholder_detail,
        real_stage,
        _,
    ) in enumerate(steps):
        real = stage >= real_stage
        detail = real_detail if real else placeholder_detail
        if name == "答案" and not real:
            detail = _answer_detail(stage)
        top = 80 + index * 92
        parts.append(_box(top, name, detail, real))
        if index + 1 < len(steps):
            parts.append(
                f'<line x1="380" y1="{top + 64}" x2="380" y2="{top + 90}" '
                'stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    parts.append(
        f'<text x="30" y="{height - 24}" '
        "font-family=\"system-ui, 'Noto Sans CJK SC', sans-serif\" "
        f'font-size="13" fill="#475569">{footer}</text>'
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
