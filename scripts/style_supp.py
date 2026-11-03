"""style_supp.py — type and colour conventions for the supplementary figures.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# The plotting helpers come from the `figure-style` runtime, which is not
# redistributed here. Either point FIGURE_STYLE_RUNTIME at that runtime.py, or
# drop a copy at scripts/figure_style_runtime.py.
# Only the figure scripts need this module; every analysis script (03-45) runs
# without it.
_RUNTIME_CANDIDATES = [
    os.environ.get('FIGURE_STYLE_RUNTIME'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'figure_style_runtime.py'),
]
for _cand in _RUNTIME_CANDIDATES:
    if _cand and os.path.exists(_cand):
        exec(compile(open(_cand, encoding='utf-8').read(), 'runtime.py', 'exec'))
        break
else:
    raise RuntimeError(
        'figure-style runtime not found. Set FIGURE_STYLE_RUNTIME to the '
        'runtime.py of the figure-style skill, or place a copy at '
        'scripts/figure_style_runtime.py. Analysis scripts do not need it.'
    )
apply_figure_style(sizes=(8, 7, 6))
mpl.rcParams.update({
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'figure.dpi': 150,
    # clinical-readability round: Arial throughout, and no text below 6.2 pt at
    # final printed width (the previous smallest label was 5.6 pt)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'xtick.labelsize': 6.2,
    'ytick.labelsize': 6.2,
    'legend.fontsize': 6.2,
})
MIN_PT = 6.2

MM = 1 / 25.4
W2 = 180 * MM          # double-column width
W1 = 88 * MM           # single-column width
FIGDIR = 'figures'

# entity -> colour (bound once, reused in every figure)
SRC = {
    'CHARLS': '#1f6fb2',   # China
    'ELSA':   '#c2571a',   # England
    'HRS':    '#2e7d5b',   # USA
    'SHARE':  '#6b5b95',   # Europe
    'POOLED': '#333333',
}
SRC_LAB = {'CHARLS': 'CHARLS (China)', 'ELSA': 'ELSA (England)',
           'HRS': 'HRS (USA)', 'SHARE': 'SHARE (Europe)', 'POOLED': 'Pooled'}

# strategy -> colour: the questionnaire tier is the focal hue
STRAT = {
    'R':  '#9aa7b1',
    'Q':  '#1f6fb2',
    'E1': '#c2571a',
    'E2': '#2e7d5b',
    'E3': '#6b5b95',
    'L':  '#8c6d31',
}
STRAT_LAB = {'R': 'R  background', 'Q': 'Q  + questionnaire',
             'E1': 'E1  Q + grip', 'E2': 'E2  Q + memory',
             'E3': 'E3  Q + grip + memory', 'L': 'L  E3 + prior-wave measures'}

# semantic greys / accents
GREY = '#8a8a8a'
DARK = '#2b2b2b'
LIGHT = '#d9e2ea'
ALERT = '#b03a2e'      # reserved for error / anomaly only
OK = '#1f6fb2'


def fig(w=W2, h=60):
    return plt.figure(figsize=(w, h * MM))


def tidy(ax, top=True, right=True):
    for sp in ('top', 'right'):
        if (sp == 'top' and top) or (sp == 'right' and right):
            ax.spines[sp].set_visible(False)
    ax.margins(0.04)


def _min_fontsize(fig):
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    return min(sizes) if sizes else None


def save(fig, name, tight=True):
    """Save PDF + PNG and run the collision and minimum-font QA passes."""
    os.makedirs(FIGDIR, exist_ok=True)
    fs = _min_fontsize(fig)
    if fs is not None and fs < MIN_PT - 0.01:
        print(f'{name}: WARNING smallest label {fs:.2f} pt < {MIN_PT} pt')
    png = f'{FIGDIR}/{name}.png'
    pdf = f'{FIGDIR}/{name}.pdf'
    kw = dict(bbox_inches='tight', pad_inches=0.02, facecolor='white')
    fig.savefig(png, dpi=300, **kw)
    fig.savefig(pdf, **kw)

    rend = fig.canvas.get_renderer()
    labels = [(t, t.get_window_extent(rend)) for t in fig.findobj(mpl.text.Text)
              if t.get_text().strip() and t.get_visible()]
    frames = [(s, s.get_window_extent(rend)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    own = {ax: set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'))
           for ax in fig.axes}
    hits = [f'{a.get_text()!r} x {b.get_text()!r}'
            for i, (a, ba) in enumerate(labels)
            for b, bb in labels[i + 1:] if ba.overlaps(bb)]
    hits += [f'{t.get_text()!r} x spine'
             for t, bt in labels for s, bs in frames
             if bt.overlaps(bs) and t not in own.get(s.axes, set())]
    status = 'CLEAN' if not hits else 'COLLISIONS'
    print(f'{name}: {status} {hits if hits else ""}')
    plt.close(fig)
    return png, pdf, hits


def ci_band(ax, x, lo, hi, color, alpha=0.18):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def diag(ax, lim, label=True):
    ax.plot([0, lim], [0, lim], color='#bbbbbb', lw=0.8, ls=(0, (3, 2)), zorder=0)
    if label:
        ax.annotate('perfect calibration', xy=(lim * 0.97, lim * 0.97),
                    xytext=(lim * 0.52, lim * 0.86), fontsize=6, color='#7a7a7a',
                    ha='center', va='center')
