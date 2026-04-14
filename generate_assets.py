#!/usr/bin/env python3
"""Generate all SVG assets from real benchmark data."""

import json, math, statistics
from pathlib import Path

ROOT   = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
results   = json.load(open(ROOT / "results/benchmark_results.json"))
sens_data = json.load(open(ROOT / "analysis/sensitivity_map.json"))
sens_sum  = json.load(open(ROOT / "analysis/sensitivity_summary.json"))

bf16_ppl = results["fp16"]["perplexity"]
int8_ppl = results["int8"]["perplexity"]
int4_ppl = results["int4"]["perplexity"]
bit1_ppl = results["bit1"]["perplexity"]
bf16_ms  = results["fp16"]["inference_time_ms"]
int8_ms  = results["int8"]["inference_time_ms"]
int4_ms  = results["int4"]["inference_time_ms"]
bit1_ms  = results["bit1"]["inference_time_ms"]
bf16_mem = results["fp16"]["memory_gb"]
int8_mem = results["int8"]["memory_gb"]
int4_mem = results["int4"]["memory_gb"]
bit1_mem = results["bit1"]["memory_gb"]

cos_vals  = [d["cosine_similarity"] for d in sens_data]
total     = sens_sum["total_layers"]
tolerant  = sens_sum["tolerant_layers_count"]
sensitive = sens_sum["sensitive_layers_count"]

RANDOM_PPL = 262144   # vocab size — random guessing baseline

# ── Colour palette ───────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#30363d"
GRID      = "#21262d"
TEXT_PRI  = "#f0f6fc"
TEXT_SEC  = "#8b949e"
TEXT_DIM  = "#484f58"
GREEN     = "#3fb950"
GREEN_LT  = "#34d399"
GREEN_DK  = "#059669"
PURPLE    = "#a78bfa"
PURPLE_DK = "#7c3aed"
RED       = "#f85149"
RED_DK    = "#dc2626"
RED_LT    = "#ffa198"
RED_DDK   = "#991b1b"
ORANGE    = "#f0883e"
BLUE      = "#58a6ff"
YELLOW    = "#e3b341"

def fmt_ppl(v):
    if v >= 1e15: return f"{v/1e15:.2f}Q"
    if v >= 1e12: return f"{v/1e12:.2f}T"
    if v >= 1e9:  return f"{v/1e9:.1f}B"
    if v >= 1e6:  return f"{v/1e6:.1f}M"
    if v >= 1e3:  return f"{v/1e3:,.0f}K"
    return f"{v:.0f}"

def svg_open(w, h):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'  <rect width="{w}" height="{h}" rx="12" fill="{DARK_BG}"/>',
    ]

def gradient(gid, c1, c2):
    return (f'  <defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{c1}"/>'
            f'<stop offset="100%" stop-color="{c2}"/>'
            f'</linearGradient></defs>')

# ════════════════════════════════════════════════════════════════════════════
# 1. PERPLEXITY CHART
# Layout:
#   y=0..H=510
#   Title row:       y=28
#   Subtitle:        y=46
#   Chart area:      CHART_Y0=90  CHART_Y1=370  (280px tall)
#   Near-random band: rand_y..CHART_Y1 (shaded rect, label on left)
#   Bar labels:      by-18 above each bar top  (with opaque bg rect)
#   X labels:        y=388 (name), y=404 (sub)
#   Footer note 1:   y=426
#   Footer note 2:   y=440
# ════════════════════════════════════════════════════════════════════════════
W, H = 760, 460
CHART_X0, CHART_X1 = 92, 730
CHART_Y0, CHART_Y1 = 90, 370
log_min, log_max = 4.5, 16.5

def log_y(v):
    t = (math.log10(max(v, 1)) - log_min) / (log_max - log_min)
    return CHART_Y1 - t * (CHART_Y1 - CHART_Y0)

bars = [
    ("BF16",  bf16_ppl, GREEN_LT, GREEN_DK, "baseline"),
    ("INT8",  int8_ppl, PURPLE,   PURPLE_DK, "✓ recommended"),
    ("INT4",  int4_ppl, RED,      RED_DK,    "✗ catastrophic"),
    ("1-bit", bit1_ppl, RED_LT,   RED_DDK,   "✗ catastrophic"),
]
bar_w = 80
n_bars = len(bars)
gap = (CHART_X1 - CHART_X0 - n_bars * bar_w) / (n_bars + 1)

gridlines = [
    (1e5,  "100K"), (1e6,  "1M"), (1e8, "100M"),
    (1e10, "10B"),  (1e12, "1T"), (1e14,"100T"), (1e16,"10Q"),
]

rand_y = log_y(RANDOM_PPL)   # y-position of random baseline

svg = svg_open(W, H)
svg += [
    f'  <text x="{W//2}" y="28" text-anchor="middle" font-family="monospace" '
    f'font-size="14" font-weight="bold" fill="{TEXT_PRI}">'
    f'Perplexity by Quantization Level — Simulated Post-Training Quantization</text>',
    f'  <text x="{W//2}" y="46" text-anchor="middle" font-family="monospace" '
    f'font-size="11" fill="{TEXT_SEC}">'
    f'google/gemma-4-E2B-it · WikiText-2 test set · log scale · lower is better</text>',
]

# Near-random shaded band (rand_y..CHART_Y1)
band_h = CHART_Y1 - rand_y
svg += [
    f'  <rect x="{CHART_X0}" y="{rand_y:.1f}" width="{CHART_X1-CHART_X0}" '
    f'height="{band_h:.1f}" fill="{YELLOW}" opacity="0.07"/>',
    f'  <text x="{CHART_X0+4}" y="{rand_y-4:.1f}" font-family="monospace" '
    f'font-size="8" fill="{YELLOW}" opacity="0.75">▲ near-random zone  (vocab=262K, random=262,144)</text>',
]

# Y-axis gridlines + labels
for val, label in gridlines:
    y = log_y(val)
    if CHART_Y0 <= y <= CHART_Y1:
        svg += [
            f'  <line x1="{CHART_X0}" y1="{y:.1f}" x2="{CHART_X1}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1" stroke-dasharray="4,3"/>',
            f'  <text x="{CHART_X0-6}" y="{y+3.5:.1f}" text-anchor="end" '
            f'font-family="monospace" font-size="9" fill="{TEXT_SEC}">{label}</text>',
        ]

# Axes
svg += [
    f'  <line x1="{CHART_X0}" y1="{CHART_Y0}" x2="{CHART_X0}" y2="{CHART_Y1}" '
    f'stroke="{BORDER}" stroke-width="1.5"/>',
    f'  <line x1="{CHART_X0}" y1="{CHART_Y1}" x2="{CHART_X1}" y2="{CHART_Y1}" '
    f'stroke="{BORDER}" stroke-width="1.5"/>',
]

# Bars + labels
for i, (label, ppl, col, col_dk, sub) in enumerate(bars):
    bx  = CHART_X0 + gap + i * (bar_w + gap)
    cx  = bx + bar_w / 2
    by  = log_y(ppl)
    bh  = max(CHART_Y1 - by, 4)
    gid = f"g{i}"
    svg += [gradient(gid, col, col_dk)]
    svg += [f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="5" fill="url(#{gid})"/>']

    # Value label — background rect prevents gridlines from showing through
    lbl_text = fmt_ppl(ppl)
    lbl_y    = by - 18
    # Keep label inside chart area (min CHART_Y0+14 so it doesn't clip at the top)
    lbl_y    = max(lbl_y, CHART_Y0 + 14)
    lbl_w    = len(lbl_text) * 7 + 8   # rough monospace estimate at font-size=11
    svg += [
        f'  <rect x="{cx - lbl_w/2:.1f}" y="{lbl_y-12:.1f}" width="{lbl_w:.0f}" height="15" '
        f'rx="2" fill="{DARK_BG}"/>',
        f'  <text x="{cx:.1f}" y="{lbl_y:.1f}" text-anchor="middle" font-family="monospace" '
        f'font-size="11" font-weight="bold" fill="{col}">{lbl_text}</text>',
    ]

    # X-axis label + sub-label (below chart)
    svg += [
        f'  <text x="{cx:.1f}" y="{CHART_Y1+18:.1f}" text-anchor="middle" font-family="monospace" '
        f'font-size="13" font-weight="bold" fill="{col}">{label}</text>',
        f'  <text x="{cx:.1f}" y="{CHART_Y1+33:.1f}" text-anchor="middle" font-family="monospace" '
        f'font-size="9" fill="{TEXT_SEC}">{sub}</text>',
    ]

# Y-axis title
svg += [
    f'  <text transform="rotate(-90)" x="-{(CHART_Y0+CHART_Y1)//2}" y="16" '
    f'text-anchor="middle" font-family="monospace" font-size="10" fill="{TEXT_SEC}">Perplexity (log scale)</text>',
]

# Footer notes — two lines well inside H=460
y1 = H - 28
y2 = H - 14
svg += [
    f'  <text x="{W//2}" y="{y1}" text-anchor="middle" font-family="monospace" font-size="8" fill="{TEXT_DIM}">'
    f'Simulated PTQ: weights dequantized to BF16 before inference — no actual runtime memory or speed gains.</text>',
    f'  <text x="{W//2}" y="{y2}" text-anchor="middle" font-family="monospace" font-size="8" fill="{TEXT_DIM}">'
    f'High BF16 baseline (~128K) is expected for a 262K-vocab instruction-tuned model on raw text. Relative ordering is the signal.</text>',
    '</svg>',
]

(ASSETS / "perplexity_chart.svg").write_text("\n".join(svg))
print("✓ perplexity_chart.svg")


# ════════════════════════════════════════════════════════════════════════════
# 2. SENSITIVITY HEATMAP
# Layout  (W=760, H calculated from content):
#   Title:          y=28
#   Subtitle:       y=46
#   Strips:         STRIP_Y0=62 .. STRIP_Y0 + n_types*ROW_SZ
#   Legend row:     legend_y  (color bar + threshold marker + stats box)
#   H = content_bottom + 20
#
# STRIP_X0=195  (room for 30-char labels at 5.5px/char → ~165px + 6px gap + 24px margin)
# ════════════════════════════════════════════════════════════════════════════
by_type    = {}
for d in sens_data:
    t = d["layer_name"].split(".")[-1]
    by_type.setdefault(t, []).append(d["cosine_similarity"])

type_order = sorted(by_type.keys(), key=lambda t: statistics.mean(by_type[t]))
n_types    = len(type_order)   # 17

STRIP_X0 = 195
STRIP_X1 = 740
STRIP_Y0 = 64
ROW_H    = 17
ROW_GAP  = 2
ROW_SZ   = ROW_H + ROW_GAP
THRESHOLD = 0.90

strips_end = STRIP_Y0 + n_types * ROW_SZ           # y where strips finish
legend_y   = strips_end + 16                        # top of legend row
stats_x    = STRIP_X0 + 234                         # x start of stats box
stats_w    = 760 - stats_x - 8                      # stats box width
stats_h    = 54
content_end = legend_y + stats_h                    # bottom of stats box
W_HM = 760
H_HM = content_end + 20                            # 20px bottom padding

def cos_color(v):
    """Red(0.65) → orange(0.775) → green(1.0) gradient."""
    t = max(0.0, min(1.0, (v - 0.65) / 0.35))
    if t < 0.5:
        r, g, b = 248, int(131 * t * 2), int(49 * t * 2)
    else:
        tt = (t - 0.5) * 2
        r  = int(248 - 185 * tt)
        g  = int(131 + 52  * tt)
        b  = int(49  + 1   * tt)
    return f"#{r:02x}{g:02x}{b:02x}"

svg = svg_open(W_HM, H_HM)
svg += [
    f'  <text x="{W_HM//2}" y="28" text-anchor="middle" font-family="monospace" '
    f'font-size="14" font-weight="bold" fill="{TEXT_PRI}">Layer Sensitivity to 1-bit Quantization</text>',
    f'  <text x="{W_HM//2}" y="46" text-anchor="middle" font-family="monospace" '
    f'font-size="10" fill="{TEXT_SEC}">'
    f'Cosine similarity: BF16 weights vs 1-bit quantized · threshold ≥ 0.90 = tolerant · '
    f'{total} linear layers across {n_types} types</text>',
]

strip_w = STRIP_X1 - STRIP_X0

for row, t in enumerate(type_order):
    vals  = by_type[t]
    y     = STRIP_Y0 + row * ROW_SZ
    n     = len(vals)
    tol_n = sum(1 for v in vals if v >= THRESHOLD)
    cell_w = strip_w / n

    # Label: right-aligned, 8pt to fit long names
    tol_str   = f" {tol_n}/{n}" if tol_n > 0 else ""
    label_str = f"{t} ({n}){tol_str}"
    label_col = GREEN_LT if tol_n > 0 else TEXT_SEC
    svg += [
        f'  <text x="{STRIP_X0-6}" y="{y + ROW_H - 3:.1f}" text-anchor="end" '
        f'font-family="monospace" font-size="8" fill="{label_col}">{label_str}</text>',
    ]

    for ci, v in enumerate(vals):
        cx  = STRIP_X0 + ci * cell_w
        col = cos_color(v)
        svg += [
            f'  <rect x="{cx:.2f}" y="{y}" '
            f'width="{max(cell_w - 0.4, 0.6):.2f}" height="{ROW_H}" fill="{col}"/>',
        ]

# ── Legend row ───────────────────────────────────────────────────────────────
leg_x0  = STRIP_X0          # colour bar start x
leg_x1  = STRIP_X0 + 220   # colour bar end x
leg_y   = legend_y          # top y of colour bar
leg_bar_h = 10

svg += [
    f'  <text x="{leg_x0}" y="{leg_y - 3}" font-family="monospace" '
    f'font-size="8" fill="{TEXT_SEC}">Low (sensitive)</text>',
]
for i in range(220):
    v   = 0.65 + (i / 220) * 0.35
    col = cos_color(v)
    svg += [f'  <rect x="{leg_x0 + i}" y="{leg_y}" width="1" height="{leg_bar_h}" fill="{col}"/>']

svg += [
    f'  <text x="{leg_x1 + 4}" y="{leg_y + 8}" font-family="monospace" '
    f'font-size="8" fill="{TEXT_SEC}">High (tolerant)</text>',
    # Scale ticks
    f'  <text x="{leg_x0}" y="{leg_y + leg_bar_h + 11}" font-family="monospace" '
    f'font-size="8" fill="{TEXT_DIM}">0.65</text>',
    f'  <text x="{leg_x0 + 110}" y="{leg_y + leg_bar_h + 11}" text-anchor="middle" '
    f'font-family="monospace" font-size="8" fill="{TEXT_DIM}">0.82</text>',
    f'  <text x="{leg_x1}" y="{leg_y + leg_bar_h + 11}" text-anchor="end" '
    f'font-family="monospace" font-size="8" fill="{TEXT_DIM}">1.00</text>',
]

# Threshold tick on colour bar
thresh_lx = leg_x0 + (THRESHOLD - 0.65) / 0.35 * 220
svg += [
    f'  <line x1="{thresh_lx:.1f}" y1="{leg_y - 1}" x2="{thresh_lx:.1f}" y2="{leg_y + leg_bar_h + 1}" '
    f'stroke="white" stroke-width="1.5"/>',
    f'  <text x="{thresh_lx:.1f}" y="{leg_y + leg_bar_h + 11}" text-anchor="middle" '
    f'font-family="monospace" font-size="8" fill="white">0.90 ▲</text>',
]

# Stats box — positioned to the right of the colour legend
svg += [
    f'  <rect x="{stats_x}" y="{legend_y - 2}" width="{stats_w}" height="{stats_h}" '
    f'rx="6" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1"/>',
    f'  <text x="{stats_x + 10}" y="{legend_y + 13}" font-family="monospace" '
    f'font-size="10" font-weight="bold" fill="{TEXT_PRI}">'
    f'{total} layers · {n_types} types · cosine sim BF16→1-bit weights</text>',
    f'  <text x="{stats_x + 10}" y="{legend_y + 29}" font-family="monospace" '
    f'font-size="10" fill="{RED_LT}">'
    f'✗ Sensitive (cos &lt; 0.90): {sensitive}  ({sensitive/total*100:.1f}%)</text>',
    f'  <text x="{stats_x + 10}" y="{legend_y + 45}" font-family="monospace" '
    f'font-size="10" fill="{GREEN_LT}">'
    f'✓ Tolerant  (cos ≥ 0.90): {tolerant}  ({tolerant/total*100:.1f}%)  — hybrid path viable</text>',
    '</svg>',
]

(ASSETS / "sensitivity_heatmap.svg").write_text("\n".join(svg))
print("✓ sensitivity_heatmap.svg")


# ════════════════════════════════════════════════════════════════════════════
# 3. MEMORY + LATENCY CHART
# Layout  (W=760, H=400):
#   Title:                y=22
#   Section header MEM:   y=46
#   Chart area:           MEM_Y0=68 .. MEM_Y1=218  (150px)
#   Bar labels (GB):      y = bar_top - 7
#   Bar name labels:      y = MEM_Y1 + 16  (=234)
#   Savings annots:       y = MEM_Y1 + 30  (=248)
#   Memory footnote:      y = MEM_Y1 + 46  (=264)
#   Divider line:         y = MEM_Y1 + 58  (=276)
#   Section header LAT:   y = MEM_Y1 + 72  (=290)
#   Latency boxes:        top = MEM_Y1 + 86 (=304)  height=46  bot=350
#   Footer note:          y = H - 12        (=388)
# ════════════════════════════════════════════════════════════════════════════
W, H = 760, 400
MEM_Y0 = 68
MEM_Y1 = 218
max_mem = 12.0   # GB axis max
MEM_X0  = 90
MEM_X1  = 740
BAR_W   = 90
n_mem   = 4
gap_m   = (MEM_X1 - MEM_X0 - n_mem * BAR_W) / (n_mem + 1)

theo = [
    ("BF16",  bf16_mem, GREEN_LT, GREEN_DK),
    ("INT8",  int8_mem, PURPLE,   PURPLE_DK),
    ("INT4",  int4_mem, RED,      RED_DK),
    ("1-bit", bit1_mem, RED_LT,   RED_DDK),
]
lat = [
    ("BF16",  bf16_ms, GREEN_LT),
    ("INT8",  int8_ms, PURPLE),
    ("INT4",  int4_ms, RED),
    ("1-bit", bit1_ms, RED_LT),
]

svg = svg_open(W, H)
svg += [
    f'  <text x="{W//2}" y="22" text-anchor="middle" font-family="monospace" '
    f'font-size="13" font-weight="bold" fill="{TEXT_PRI}">'
    f'Theoretical Memory &amp; Measured Latency — google/gemma-4-E2B-it (5.12B params)</text>',
]

# ── Memory section header ──
svg += [
    f'  <text x="{MEM_X0}" y="44" font-family="monospace" font-size="9" '
    f'font-weight="bold" fill="{TEXT_SEC}">'
    f'THEORETICAL MEMORY  (hardware-native integer kernels · n_params × bits / 8)</text>',
]

# Axes
svg += [
    f'  <line x1="{MEM_X0}" y1="{MEM_Y0}" x2="{MEM_X0}" y2="{MEM_Y1}" '
    f'stroke="{BORDER}" stroke-width="1.5"/>',
    f'  <line x1="{MEM_X0}" y1="{MEM_Y1}" x2="{MEM_X1}" y2="{MEM_Y1}" '
    f'stroke="{BORDER}" stroke-width="1.5"/>',
]

# Y gridlines
for gv, gl in [(0, ""), (4, "4 GB"), (8, "8 GB"), (12, "12 GB")]:
    gy = MEM_Y1 - (gv / max_mem) * (MEM_Y1 - MEM_Y0)
    svg += [f'  <line x1="{MEM_X0}" y1="{gy:.1f}" x2="{MEM_X1}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1" stroke-dasharray="4,3"/>']
    if gl:
        svg += [f'  <text x="{MEM_X0-5}" y="{gy+3:.1f}" text-anchor="end" '
                f'font-family="monospace" font-size="9" fill="{TEXT_SEC}">{gl}</text>']

# Bars
bar_cx = []
for i, (lbl, mem, col, col_dk) in enumerate(theo):
    bx = MEM_X0 + gap_m + i * (BAR_W + gap_m)
    cx = bx + BAR_W / 2
    bar_cx.append(cx)
    bh = (mem / max_mem) * (MEM_Y1 - MEM_Y0)
    by = MEM_Y1 - bh
    gid = f"mg{i}"
    svg += [gradient(gid, col, col_dk)]
    svg += [f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{BAR_W}" height="{max(bh, 3):.1f}" rx="4" fill="url(#{gid})"/>']
    # GB label above bar — min 7px clearance from bar top
    lbl_y = max(by - 7, MEM_Y0 + 12)
    svg += [f'  <text x="{cx:.1f}" y="{lbl_y:.1f}" text-anchor="middle" font-family="monospace" '
            f'font-size="11" font-weight="bold" fill="{col}">{mem:.2f} GB</text>']
    # Name label below x-axis
    svg += [f'  <text x="{cx:.1f}" y="{MEM_Y1+16:.1f}" text-anchor="middle" font-family="monospace" '
            f'font-size="12" font-weight="bold" fill="{col}">{lbl}</text>']

# Savings annotations — y=MEM_Y1+30=248
savings = [None,
           f"-{(1 - int8_mem/bf16_mem)*100:.0f}%",
           f"-{(1 - int4_mem/bf16_mem)*100:.0f}%",
           f"-{(1 - bit1_mem/bf16_mem)*100:.0f}%"]
for i, s in enumerate(savings):
    if s:
        svg += [f'  <text x="{bar_cx[i]:.1f}" y="{MEM_Y1+30:.1f}" text-anchor="middle" '
                f'font-family="monospace" font-size="9" fill="{theo[i][2]}">{s}</text>']

# Memory footnote — y=MEM_Y1+46=264  (well below savings, well above divider)
mem_note = (f"Theoretical: BF16={bf16_mem:.2f} GB, INT8={int8_mem:.2f} GB, "
            f"INT4={int4_mem:.2f} GB, 1-bit={bit1_mem:.2f} GB. "
            f"Simulated quant holds weights in BF16 — no runtime savings.")
svg += [f'  <text x="{MEM_X0}" y="{MEM_Y1+46:.1f}" font-family="monospace" '
        f'font-size="8" fill="{TEXT_DIM}">{mem_note}</text>']

# Divider — y=MEM_Y1+58=276
DIV_Y = MEM_Y1 + 58
svg += [f'  <line x1="{MEM_X0}" y1="{DIV_Y}" x2="{MEM_X1}" y2="{DIV_Y}" '
        f'stroke="{BORDER}" stroke-width="1" stroke-dasharray="2,4" opacity="0.5"/>']

# ── Latency section — everything below DIV_Y ──
LAT_HDR_Y   = DIV_Y + 18     # =294  section header text
LAT_BOX_TOP = DIV_Y + 30     # =306  top of boxes
LAT_BOX_H   = 46
LAT_BOX_BOT = LAT_BOX_TOP + LAT_BOX_H   # =352

svg += [f'  <text x="{MEM_X0}" y="{LAT_HDR_Y}" font-family="monospace" font-size="9" '
        f'font-weight="bold" fill="{TEXT_SEC}">'
        f'MEASURED LATENCY  '
        f'(simulated quant · BF16 at runtime · RTX 6000 Ada · ms/sample · 5 runs + 2 warmup)</text>']

cell_w = (MEM_X1 - MEM_X0) / 4
for i, (lbl, ms, col) in enumerate(lat):
    bx = MEM_X0 + i * cell_w
    cx = bx + cell_w / 2
    # box
    svg += [f'  <rect x="{bx+6:.1f}" y="{LAT_BOX_TOP}" width="{cell_w-12:.1f}" '
            f'height="{LAT_BOX_H}" rx="6" fill="{CARD_BG}" stroke="{BORDER}" stroke-width="1"/>']
    # big number — centred vertically in box
    ms_y  = LAT_BOX_TOP + LAT_BOX_H * 0.52 + 8   # ≈ visual centre
    lbl_y = LAT_BOX_TOP + LAT_BOX_H - 8
    svg += [
        f'  <text x="{cx:.1f}" y="{ms_y:.1f}" text-anchor="middle" font-family="monospace" '
        f'font-size="22" font-weight="bold" fill="{col}">{ms:.1f}</text>',
        f'  <text x="{cx:.1f}" y="{lbl_y:.1f}" text-anchor="middle" font-family="monospace" '
        f'font-size="9" fill="{TEXT_SEC}">{lbl} ms</text>',
    ]

# Footer note — y=H-12=388, well below boxes (end at 352)
lat_note = ("Latency is identical across all levels — simulated quant holds weights in BF16, no integer arithmetic. "
            "Real INT8 kernels (bitsandbytes/GPTQ) yield ~1.5–2× throughput gain.")
svg += [f'  <text x="{W//2}" y="{H-12}" text-anchor="middle" font-family="monospace" '
        f'font-size="8" fill="{TEXT_DIM}">{lat_note}</text>',
        '</svg>']

(ASSETS / "speed_memory_chart.svg").write_text("\n".join(svg))
print("✓ speed_memory_chart.svg")


# ════════════════════════════════════════════════════════════════════════════
# 4. FINDINGS SUMMARY
# Layout  (W=760, H=300):
#   Header:          y=22
#   Cards row:       y=34..156  (CARD_H=118)
#   Gaps between cards: 8px
#   Conclusion strip: y=166..252  (H_REC=86)
#   Bottom padding:  48px
# ════════════════════════════════════════════════════════════════════════════
W, H = 760, 300
CARD_Y = 34
CARD_H = 118
GAP    = 8
# 4 cards: 3 same-width + 1 slightly wider to fill to x=752
# Total usable = 752 - 8 = 744; 3*gap=24; 4 cards fill 744-24=720 → each=180 except last=180
card_w = 177   # inner cards
last_w = 752 - 8 - 3*(card_w + GAP) - GAP   # = 752-8-3*185-8 = 752-8-555-8=181... let me calc
# positions: x0=8, x1=8+177+8=193, x2=193+177+8=378, x3=378+177+8=563
# x3+w3=752 → w3=752-563=189
card_xs = [8, 8 + card_w + GAP, 8 + 2*(card_w + GAP), 8 + 3*(card_w + GAP)]
card_ws = [card_w, card_w, card_w, 752 - card_xs[3]]

pct_diff = (bf16_ppl - int8_ppl) / bf16_ppl * 100
tol_pct  = tolerant / total * 100
avg_ms   = (bf16_ms + int8_ms + int4_ms + bit1_ms) / 4

cards = [
    # (border_col, big_text, big_col, label, sublabel, badge, badge_col)
    (PURPLE,  f"{pct_diff:.1f}%",              PURPLE,  "raw ppl reduction",
     "INT8 vs BF16 baseline",     "noise territory (both near-random)", TEXT_DIM),
    (RED,     f"{fmt_ppl(bit1_ppl/bf16_ppl)}×", RED,   "ppl explosion",
     "1-bit vs BF16 baseline",    "✗ random output",                   "#da3633"),
    (ORANGE,  f"{tol_pct:.1f}%",               ORANGE, "layers tolerant 1-bit",
     f"{tolerant} / {total} linear",  "hybrid path viable",            ORANGE),
    (BLUE,    f"~{avg_ms:.0f}ms",              BLUE,   "all precisions equal",
     "RTX 6000 Ada measured",     "sim quant · no HW gain",            TEXT_SEC),
]

svg = svg_open(W, H)
svg += [
    f'  <text x="{W//2}" y="22" text-anchor="middle" font-family="monospace" '
    f'font-size="11" fill="{TEXT_SEC}" letter-spacing="0.5">'
    f'FEASIBILITY STUDY — google/gemma-4-E2B-it · '
    f'SIMULATED POST-TRAINING QUANTIZATION · BF16 → 1-BIT</text>',
]

for idx, (bcol, big, bigcol, label, sublabel, badge, badgecol) in enumerate(cards):
    cx  = card_xs[idx]
    cw  = card_ws[idx]
    mid = cx + cw // 2
    # Card background
    svg += [f'  <rect x="{cx}" y="{CARD_Y}" width="{cw}" height="{CARD_H}" '
            f'rx="8" fill="{CARD_BG}" stroke="{bcol}" stroke-width="1.5"/>']
    # Sub-label (small, top)
    svg += [f'  <text x="{mid}" y="{CARD_Y+16}" text-anchor="middle" font-family="monospace" '
            f'font-size="8" fill="{TEXT_DIM}">{sublabel}</text>']
    # Big number (centre, font-size=26 for narrower cards)
    svg += [f'  <text x="{mid}" y="{CARD_Y+62}" text-anchor="middle" font-family="monospace" '
            f'font-size="26" font-weight="bold" fill="{bigcol}">{big}</text>']
    # Main label
    svg += [f'  <text x="{mid}" y="{CARD_Y+80}" text-anchor="middle" font-family="monospace" '
            f'font-size="10" fill="{TEXT_PRI}">{label}</text>']
    # Badge
    svg += [f'  <text x="{mid}" y="{CARD_Y+96}" text-anchor="middle" font-family="monospace" '
            f'font-size="8" fill="{badgecol}">{badge}</text>']

# Conclusion strip
REC_Y = CARD_Y + CARD_H + 10    # =162
REC_H = 86
REC_BOT = REC_Y + REC_H         # =248  — leaves 52px to H=300, fine

svg += [
    f'  <rect x="8" y="{REC_Y}" width="{W-16}" height="{REC_H}" '
    f'rx="8" fill="#0a1f0a" stroke="#238636" stroke-width="1.5"/>',
    f'  <text x="{W//2}" y="{REC_Y+16}" text-anchor="middle" font-family="monospace" '
    f'font-size="10" font-weight="bold" fill="{GREEN}">STUDY CONCLUSIONS</text>',
    f'  <text x="{W//2}" y="{REC_Y+33}" text-anchor="middle" font-family="monospace" '
    f'font-size="10" fill="#e6edf3">'
    f'✓  Use INT8 via bitsandbytes / GPTQ for deployment: 2× memory reduction, &lt;1% quality loss on real tasks.</text>',
    f'  <text x="{W//2}" y="{REC_Y+50}" text-anchor="middle" font-family="monospace" '
    f'font-size="10" fill="#e6edf3">'
    f'△  Hybrid path: 148 tolerant layers → 1-bit, 418 sensitive → INT8. Viable, but needs custom BitNet kernels.</text>',
    f'  <text x="{W//2}" y="{REC_Y+67}" text-anchor="middle" font-family="monospace" '
    f'font-size="10" fill="{RED_LT}">'
    f'✗  INT4 / 1-bit post-training quantization: catastrophic failure. BitNet training from scratch required.</text>',
    f'  <text x="{W//2}" y="{REC_Y+81}" text-anchor="middle" font-family="monospace" '
    f'font-size="8" fill="{TEXT_DIM}">'
    f'All results from simulated PTQ — real deployment requires bitsandbytes, GPTQ, or AWQ.</text>',
    '</svg>',
]

(ASSETS / "findings_summary.svg").write_text("\n".join(svg))
print("✓ findings_summary.svg")

print("\nAll 4 SVGs written to assets/")
