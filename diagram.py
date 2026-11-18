"""
SVG drawings of a model's real architecture, with QLoRA adapters overlaid.

Pure string building -- no plotting library -- so the diagrams are crisp at any
zoom, render inside Gradio's HTML component, and work in light and dark mode
(text uses currentColor; fills are translucent).
"""

from __future__ import annotations

from html import escape

from inspect_model import Leaf, ModelInfo, human

ACCENT = "#f97316"      # LoRA orange
ATTN = "rgba(59,130,246,0.16)"
MLP = "rgba(16,185,129,0.16)"
NORM = "rgba(148,163,184,0.22)"
OTHER = "rgba(168,85,247,0.14)"
STROKE = "rgba(128,128,128,0.55)"
FONT = "font-family:ui-sans-serif,system-ui,sans-serif"


def _wrap(svg: str, w: int, h: int) -> str:
    return (f'<div style="overflow-x:auto"><svg viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px;color:var(--body-text-color,#222);{FONT}" '
            f'xmlns="http://www.w3.org/2000/svg">{svg}</svg></div>')


def _box(x, y, w, h, fill, label, sub="", stroke=STROKE, sw=1, tip="", rx=8, size=13):
    t = f"<title>{escape(tip)}</title>" if tip else ""
    s = (f'<g>{t}<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="{sw}"/>')
    ty = y + h / 2 + (-4 if sub else 4)
    s += (f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{size}" '
          f'font-weight="600" fill="currentColor">{escape(label)}</text>')
    if sub:
        s += (f'<text x="{x + w / 2}" y="{ty + 15}" text-anchor="middle" font-size="11" '
              f'fill="currentColor" opacity="0.7">{escape(sub)}</text>')
    return s + "</g>"


def _arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{STROKE}" stroke-width="1.5" '
            f'marker-end="url(#arr)"/>')


_DEFS = ('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{STROKE}"/></marker></defs>')


# --------------------------------------------------------------------------- #
# 1. Whole model: tokens -> embedding -> N x block -> norm -> head
# --------------------------------------------------------------------------- #
def model_stack(info: ModelInfo, targets: list[str] | None = None) -> str:
    targets = targets or []
    W, cx, bw = 560, 280, 380
    y = 16
    s = _DEFS
    s += f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="12" opacity="0.7" fill="currentColor">input text → token ids</text>'
    y += 10
    s += _arrow(cx, y, cx, y + 18); y += 22
    pct = lambda n: f"{100 * n / info.total_params:.1f}% of params"
    s += _box(cx - bw / 2, y, bw, 48, OTHER, f"Embedding  ({info.vocab:,} words × {info.hidden})",
              f"{human(info.embed_params)} params · {pct(info.embed_params)}",
              tip="Turns each token id into a vector of size hidden.")
    y += 48
    s += _arrow(cx, y, cx, y + 18); y += 22

    # The repeated decoder blocks, drawn as a stack of thin slabs.
    n = info.num_layers
    slab_h = 6 if n <= 40 else 4
    stack_h = n * slab_h + 64
    s += (f'<rect x="{cx - bw / 2}" y="{y}" width="{bw}" height="{stack_h}" rx="10" '
          f'fill="none" stroke="{STROKE}" stroke-dasharray="4 3"/>')
    s += (f'<text x="{cx}" y="{y + 20}" text-anchor="middle" font-size="13" font-weight="600" '
          f'fill="currentColor">Decoder block × {n}</text>')
    s += (f'<text x="{cx}" y="{y + 36}" text-anchor="middle" font-size="11" opacity="0.7" '
          f'fill="currentColor">{human(info.block_params)} params each · '
          f'{pct(info.block_params * n)} total</text>')
    yy = y + 46
    for i in range(n):
        s += (f'<g><title>layer {i}</title><rect x="{cx - bw / 2 + 24}" y="{yy}" width="{bw - 48}" '
              f'height="{slab_h - 1.5}" rx="1.5" fill="{ATTN if i % 2 else MLP}"/></g>')
        if targets:
            s += (f'<rect x="{cx + bw / 2 - 22}" y="{yy}" width="8" height="{slab_h - 1.5}" '
                  f'fill="{ACCENT}"/>')
        yy += slab_h
    if targets:
        s += (f'<text x="{cx + bw / 2 + 8}" y="{y + stack_h / 2}" font-size="11" fill="{ACCENT}" '
              f'font-weight="600">LoRA in</text><text x="{cx + bw / 2 + 8}" y="{y + stack_h / 2 + 14}" '
              f'font-size="11" fill="{ACCENT}" font-weight="600">every layer</text>')
    y += stack_h
    s += _arrow(cx, y, cx, y + 18); y += 22
    s += _box(cx - bw / 2, y, bw, 36, NORM, "Final norm", tip="Normalises the last hidden state.")
    y += 36
    s += _arrow(cx, y, cx, y + 18); y += 22
    head_sub = ("shares weights with the Embedding (tied)" if info.tied_embeddings
                else f"{human(info.head_params)} params · {pct(info.head_params)}")
    s += _box(cx - bw / 2, y, bw, 48, OTHER, f"LM head  ({info.hidden} → {info.vocab:,})", head_sub,
              tip="Scores every word in the vocabulary as the possible next token.")
    y += 48
    s += _arrow(cx, y, cx, y + 18); y += 30
    s += (f'<text x="{cx}" y="{y}" text-anchor="middle" font-size="12" opacity="0.7" '
          f'fill="currentColor">probabilities for the next token</text>')
    return _wrap(s, W, y + 12)


# --------------------------------------------------------------------------- #
# 2. Inside one decoder block
# --------------------------------------------------------------------------- #
def _group_rank(group: str, kind: str) -> int:
    g = group.lower()
    if kind == "norm":
        if any(k in g for k in ("input", "ln_1", "attention_norm", "pre_attn", "attn_norm")):
            return 0
        if "post_attention" in g:
            return 2
        if any(k in g for k in ("pre_feedforward", "ln_2", "ffn_norm", "pre_mlp")):
            return 3
        if any(k in g for k in ("post_feedforward", "post_mlp")):
            return 5
        return 6
    if "attn" in g or "attention" in g:
        return 1
    if any(k in g for k in ("mlp", "ffn", "feed_forward", "moe", "expert")):
        return 4
    return 6


def _groups(info: ModelInfo) -> list[tuple[str, list[Leaf]]]:
    order: dict[str, list[Leaf]] = {}
    for leaf in info.leaves:
        order.setdefault(leaf.group, []).append(leaf)
    items = list(order.items())
    kind = lambda leaves: "norm" if all(l.kind == "norm" for l in leaves) else "compute"
    items.sort(key=lambda kv: _group_rank(kv[0], kind(kv[1])))
    return items


def block_detail(info: ModelInfo, targets: list[str] | None = None, r: int = 16) -> str:
    targets = set(targets or [])
    W = 780
    left, right = 70, W - 20
    inner = right - left
    y = 20
    s = _DEFS
    s += (f'<text x="{W / 2}" y="{y}" text-anchor="middle" font-size="14" font-weight="700" '
          f'fill="currentColor">One decoder block ({info.layer_path}[0]) — repeated {info.num_layers}×</text>')
    y += 14
    s += (f'<text x="{W / 2}" y="{y + 6}" text-anchor="middle" font-size="11" opacity="0.65" '
          f'fill="currentColor">hover any box for details · orange = trainable LoRA adapter</text>')
    y += 22
    res_x = 34
    res_top = y
    adds: list[float] = []

    for group, leaves in _groups(info):
        is_norm = all(l.kind == "norm" for l in leaves)
        if is_norm:
            l = leaves[0]
            s += _arrow(left + inner / 2, y, left + inner / 2, y + 14); y += 16
            s += _box(left + inner / 2 - 170, y, 340, 28, NORM, f"{group}  (norm, {l.params:,} params)",
                      tip=f"{group}: {l.shape} — rescales activations, keeps training stable", size=12)
            y += 28
            continue

        g = group.lower()
        fill = ATTN if ("attn" in g or "attention" in g) else MLP if any(
            k in g for k in ("mlp", "ffn", "feed", "moe", "expert")) else OTHER
        linears = [l for l in leaves if l.kind == "linear"]
        others = [l for l in leaves if l.kind != "linear"]
        per_row = 4
        rows = max(1, -(-len(linears) // per_row)) + (1 if others else 0)
        cell_h, gap = 58, 30
        box_h = 34 + rows * (cell_h + gap)
        s += _arrow(left + inner / 2, y, left + inner / 2, y + 14); y += 16
        gsum = sum(l.params for l in leaves)
        title = {1: "Attention — tokens look at each other", 4: "MLP — per-token processing"}.get(
            _group_rank(group, "compute"), group)
        s += (f'<rect x="{left}" y="{y}" width="{inner}" height="{box_h}" rx="12" fill="{fill}" '
              f'stroke="{STROKE}"/>')
        s += (f'<text x="{left + 12}" y="{y + 20}" font-size="13" font-weight="700" fill="currentColor">'
              f'{escape(group)}</text><text x="{left + 12 + 9 * len(group) + 12}" y="{y + 20}" font-size="11" '
              f'opacity="0.7" fill="currentColor">{escape(title)} · {human(gsum)} params</text>')
        cy = y + 34
        cw = (inner - 24 - (per_row - 1) * 12) / per_row
        for i, l in enumerate(linears):
            row, col = divmod(i, per_row)
            n_in_row = min(per_row, len(linears) - row * per_row)
            row_w = n_in_row * cw + (n_in_row - 1) * 12
            bx = left + (inner - row_w) / 2 + col * (cw + 12)
            by = cy + row * (cell_h + gap)
            hit = l.short in targets
            tip = (f"{l.name}: Linear {l.in_features} → {l.out_features}\n"
                   f"{l.params:,} frozen params (4-bit in QLoRA)")
            if hit:
                lp = r * (l.in_features + l.out_features)
                tip += (f"\n+ LoRA: A ({r}×{l.in_features}) and B ({l.out_features}×{r})"
                        f" = {lp:,} trainable params ({100 * lp / l.params:.2f}% of this layer)")
            s += _box(bx, by, cw, cell_h - 18, "rgba(255,255,255,0.35)", l.short,
                      f"{l.in_features} → {l.out_features}", stroke=ACCENT if hit else STROKE,
                      sw=2.5 if hit else 1, tip=tip, size=12)
            if hit:
                # little A/B adapter drawn hanging off the frozen weight
                ay = by + cell_h - 14
                s += (f'<g><title>{escape(tip)}</title>'
                      f'<rect x="{bx + cw / 2 - 34}" y="{ay}" width="68" height="18" rx="9" fill="{ACCENT}"/>'
                      f'<text x="{bx + cw / 2}" y="{ay + 13}" text-anchor="middle" font-size="10" '
                      f'font-weight="700" fill="white">+LoRA r={r}</text></g>')
            else:
                s += (f'<text x="{bx + cw / 2}" y="{by + cell_h - 4}" text-anchor="middle" font-size="10" '
                      f'opacity="0.55" fill="currentColor">{human(l.params)} frozen</text>')
        if others:
            by = cy + (rows - 1) * (cell_h + gap)
            names = ", ".join(f"{o.name.split('.', 1)[-1]}" for o in others)
            s += (f'<text x="{left + inner / 2}" y="{by + 20}" text-anchor="middle" font-size="11" '
                  f'opacity="0.7" fill="currentColor">also: {escape(names[:110])}</text>')
        y += box_h
        adds.append(y + 16)
        s += _arrow(left + inner / 2, y, left + inner / 2, y + 8)
        s += (f'<circle cx="{left + inner / 2}" cy="{y + 16}" r="8" fill="none" stroke="{STROKE}"/>'
              f'<text x="{left + inner / 2}" y="{y + 20.5}" text-anchor="middle" font-size="13" '
              f'fill="currentColor">+</text>')
        y += 24

    # residual stream line on the left, feeding each "+"
    s += (f'<line x1="{res_x}" y1="{res_top}" x2="{res_x}" y2="{y}" stroke="{STROKE}" '
          f'stroke-width="2"/>')
    s += (f'<text x="{res_x - 6}" y="{res_top + 4}" font-size="10" opacity="0.7" fill="currentColor" '
          f'transform="rotate(-90 {res_x - 6} {res_top + 4})" text-anchor="end">residual stream</text>')
    for ay in adds:
        s += _arrow(res_x, ay, left + inner / 2 - 9, ay)
    s += _arrow(left + inner / 2, y - 8, left + inner / 2, y + 12)
    s += (f'<text x="{left + inner / 2}" y="{y + 26}" text-anchor="middle" font-size="11" opacity="0.7" '
          f'fill="currentColor">to the next block</text>')
    return _wrap(s, W, y + 36)


# --------------------------------------------------------------------------- #
# 3. After training: how much did each adapter change its layer?
# --------------------------------------------------------------------------- #
def delta_heatmap(norms: dict[tuple[int, str], float]) -> str:
    """norms[(layer, short_name)] = ||B @ A|| * scale."""
    if not norms:
        return "<p>No adapters found.</p>"
    layers = sorted({k[0] for k in norms})
    mods = sorted({k[1] for k in norms})
    hi = max(norms.values()) or 1.0
    cw, ch, lx, ty = 70, 14, 46, 50
    W = max(lx + cw * len(mods) + 20, 560)
    H = ty + ch * len(layers) + 30
    s = (f'<text x="{lx}" y="16" font-size="13" font-weight="700" fill="currentColor">'
         f'How much did training change each layer?  (‖ΔW‖ = ‖B·A‖·α/r)</text>')
    for j, m in enumerate(mods):
        s += (f'<text x="{lx + j * cw + cw / 2}" y="{ty - 8}" text-anchor="middle" font-size="11" '
              f'fill="currentColor">{escape(m)}</text>')
    for i, layer in enumerate(layers):
        y = ty + i * ch
        if i % max(1, len(layers) // 12) == 0:
            s += (f'<text x="{lx - 6}" y="{y + ch - 3}" text-anchor="end" font-size="10" '
                  f'fill="currentColor" opacity="0.7">L{layer}</text>')
        for j, m in enumerate(mods):
            v = norms.get((layer, m), 0.0)
            a = 0.08 + 0.92 * (v / hi)
            s += (f'<rect x="{lx + j * cw + 1}" y="{y}" width="{cw - 2}" height="{ch - 1}" '
                  f'fill="{ACCENT}" fill-opacity="{a:.3f}"><title>layer {layer} · {m}: {v:.4f}</title></rect>')
    s += (f'<text x="{lx}" y="{H - 8}" font-size="11" opacity="0.7" fill="currentColor">'
          f'darker = bigger change · max {hi:.4f}</text>')
    return _wrap(s, W, H)
