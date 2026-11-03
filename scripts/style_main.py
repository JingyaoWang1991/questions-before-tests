"""style_main.py — type and colour conventions for the main display figures.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- tokens
INK = '#1A1A1A'
INK2 = '#4D4D4D'
MUTED = '#7A7A7A'
HAIRLINE = '#C9C9C9'
GRID = '#DCDCDC'
SLATE = '#4F5D75'
PALE = '#C3CDD6'
GAIN = '#9E9E9E'

COH = {'CHARLS': '#3C6E9E', 'ELSA': '#C0803F', 'HRS': '#4E8F7C',
       'POOLED': '#2B2B2B', 'SHARE': '#9E9E9E'}
MK = {'CHARLS': 'o', 'ELSA': 's', 'HRS': '^', 'POOLED': 'D', 'SHARE': 'P'}
CLAB = {'CHARLS': 'China', 'ELSA': 'England', 'HRS': 'USA',
        'POOLED': 'Pooled', 'SHARE': 'Europe'}


def tint(hex_colour, frac):
    """Lighter member of one cohort's colour family.

    Mixes `hex_colour` toward white by `frac` (0 = the token itself, 1 = white),
    so a two-state comparison can use one hue at two depths instead of bringing
    in a second hue. The result is derived from a palette token, never invented,
    and each figure script prints the exact hex so it is on record.
    """
    hex_colour = hex_colour.lstrip('#')
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (0, 2, 4))
    mix = tuple(round(c + (255 - c) * frac) for c in (r, g, b))
    return '#%02X%02X%02X' % mix

def shade(hex_colour, frac):
    """Darker member of one cohort's colour family.

    Mixes `hex_colour` toward black by `frac` (0 = the token itself), the mirror
    of `tint`, so a two-state bars-in-bars comparison stays in one hue family.
    """
    hex_colour = hex_colour.lstrip('#')
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (0, 2, 4))
    mix = tuple(round(c * (1 - frac)) for c in (r, g, b))
    return '#%02X%02X%02X' % mix


def _linear(channel):
    value = channel / 255
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_colour):
    hex_colour = hex_colour.lstrip('#')
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def label_colour(background):
    """White or ink for text on a fill, whichever has the higher contrast."""
    lum = luminance(background)
    on_white = 1.05 / (lum + 0.05)
    on_ink = (lum + 0.05) / (luminance(INK) + 0.05)
    return 'white' if on_white >= on_ink else INK

# ---------------------------------------------------------------- type ladder
PT_PANEL = 12
PT_TITLE = 9.0
PT_ROW = 8.5
PT_AXIS = 8.5
PT_TICK = 7.5
PT_LEG = 7.5
PT_VAL = 7.5
PT_VAL_SM = 7.0
PT_BASE = 8.0

MM = 1 / 25.4
W2 = 170 * MM          # design width for all three figures
FIGDIR = 'figures'
CSVDIR = 'figures'

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
    'font.size': PT_BASE,
    'axes.titlesize': PT_TITLE,
    'axes.labelsize': PT_AXIS,
    'xtick.labelsize': PT_TICK,
    'ytick.labelsize': PT_TICK,
    'legend.fontsize': PT_LEG,
    'axes.linewidth': 0.9,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.major.size': 3.2,
    'ytick.major.size': 3.2,
    'pdf.compression': 6,
    'savefig.dpi': 300,
    'figure.dpi': 200,
    'lines.linewidth': 1.0,
    'hatch.linewidth': 0.8,
})


# ---------------------------------------------------------------- helpers
def apply_figure_style():
    """Explicitly apply the existing final-size role-mapped style."""
    mpl.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'xtick.direction': 'out', 'ytick.direction': 'out',
                         'savefig.dpi': 300})


def canvas(w_mm, h_mm):
    apply_figure_style()
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def clean(ax, left=True, bottom=True, top=False, right=False):
    for sp, on in (('left', left), ('bottom', bottom), ('top', top),
                   ('right', right)):
        ax.spines[sp].set_visible(on)
    for sp in ('top', 'right'):
        if ax.spines[sp].get_visible():
            ax.spines[sp].set_linewidth(0.9)
            ax.spines[sp].set_color(INK)
    ax.tick_params(axis='both', which='major', labelsize=PT_TICK,
                   direction='out', length=3.2, width=0.75, color=INK,
                   labelcolor=INK)


def panel_letter(ax, letter, dx=-0.06, dy=1.03):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=PT_PANEL,
            fontweight='bold', color=INK, ha='left', va='bottom')


def zero_line(ax, orientation='v'):
    kw = dict(color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    if orientation == 'v':
        ax.axvline(0, **kw)
    else:
        ax.axhline(0, **kw)


def ref_line(ax, value, orientation='v', label=None, label_side='top'):
    """Ink dashed reference line (never red: a reference is not a threshold)."""
    kw = dict(color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    if orientation == 'v':
        ax.axvline(value, **kw)
    else:
        ax.axhline(value, **kw)
    if label:
        if orientation == 'v':
            ax.annotate(label, xy=(value, 1.0), xycoords=('data', 'axes fraction'),
                        xytext=(0, 2), textcoords='offset points',
                        fontsize=PT_VAL_SM, color=MUTED, ha='right', va='bottom')
        else:
            ax.annotate(label, xy=(1.0, value), xycoords=('axes fraction', 'data'),
                        xytext=(-2, 3), textcoords='offset points',
                        fontsize=PT_VAL_SM, color=MUTED, ha='right', va='bottom')


def errbar(ax, x, y, lo, hi, color, marker=None, ms=5.0, lw=1.0, zorder=4,
           hollow=False, mfc=None):
    """Symmetric-looking CI drawn as a single interval plus its end caps."""
    if marker is None:
        marker = 'o'
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='butt',
            zorder=zorder)
    for xv in (lo, hi):
        ax.plot([xv, xv], [y - 0.055, y + 0.055], color=color, lw=0.9,
                zorder=zorder)
    if x is not None:
        ax.plot(x, y, marker, ms=ms, color='white' if hollow else color,
                markeredgecolor=color, markeredgewidth=0.5 if hollow else 0.5,
                zorder=zorder + 1)


# ---------------------------------------------------------------- QA + export
def _drawn_texts(fig):
    """Every text that is actually drawn, with its bbox.

    Matplotlib creates placeholder Text objects for unused tick positions; they
    report a 1x1 px bbox at the origin and are never rendered. They are skipped,
    and the text list is snapshotted before any tick accessor is called (calling
    get_ticklabels() itself creates ticks, which would otherwise be scanned).
    """
    rend = fig.canvas.get_renderer()
    out = []
    for t in list(fig.findobj(mpl.text.Text)):
        if not t.get_text().strip() or not t.get_visible():
            continue
        owner = t.axes
        if owner is not None and not owner.axison:
            continue
        try:
            bb = t.get_window_extent(rend)
        except Exception:
            continue
        if bb.width <= 1.0 and bb.height <= 1.0:
            continue
        out.append((t, bb))
    return out


def _text_overlaps(fig, tol=0.6):
    """Pairwise intersection of every drawn text bbox. Returns list of pairs."""
    items = _drawn_texts(fig)
    hits = []
    for i, (a, ba) in enumerate(items):
        for b, bb in items[i + 1:]:
            if not ba.overlaps(bb):
                continue
            w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if w > tol and h > tol:
                hits.append(f'{a.get_text()!r} x {b.get_text()!r}')
    return hits


def _artists_outside(fig, tol=0.5):
    """Anything that would be clipped: outside the exported (tight) bbox."""
    rend = fig.canvas.get_renderer()
    # get_tightbbox returns inches; artist extents are in pixels, so convert
    t_in = fig.get_tightbbox(rend)
    dpi = fig.dpi
    from matplotlib.transforms import Bbox
    tight = Bbox.from_extents(t_in.x0 * dpi, t_in.y0 * dpi,
                              t_in.x1 * dpi, t_in.y1 * dpi)
    # bbox_inches='tight' exports the union of the figure and the tight box, so
    # only artists that fall outside that union can actually be clipped.
    tight = Bbox.union([tight, fig.bbox])
    bad = []
    for arts in list(fig.findobj(mpl.artist.Artist)):
        if not getattr(arts, 'get_visible', None) or not arts.get_visible():
            continue
        if not isinstance(arts, (mpl.lines.Line2D, mpl.patches.Patch,
                                 mpl.collections.Collection)):
            continue
        if arts is getattr(fig, 'patch', None):
            continue
        owner = getattr(arts, 'axes', None)
        if owner is not None and not owner.axison:
            continue
        try:
            bb = arts.get_window_extent(rend)
        except Exception:
            continue
        if (bb.width <= 1.0 and bb.height <= 1.0):
            continue
        if (bb.x0 < tight.x0 - tol or bb.y0 < tight.y0 - tol
                or bb.x1 > tight.x1 + tol or bb.y1 > tight.y1 + tol):
            bad.append(type(arts).__name__)
    bad += [f'text {t.get_text()!r}' for t, bb in _drawn_texts(fig)
            if (bb.x0 < tight.x0 - tol or bb.y0 < tight.y0 - tol
                or bb.x1 > tight.x1 + tol or bb.y1 > tight.y1 + tol)]
    return bad


def _endpoints_outside(fig):
    """Every finite data point of every Line2D must sit inside its own axes."""
    bad = []
    for ax in fig.axes:
        # limits may be inverted (e.g. a top-down tier axis), so normalise first
        xa, xb = ax.get_xlim()
        ya, yb = ax.get_ylim()
        x0, x1 = min(xa, xb), max(xa, xb)
        y0, y1 = min(ya, yb), max(ya, yb)
        span_x, span_y = x1 - x0, y1 - y0
        for ln in ax.lines:
            xd, yd = ln.get_xdata(), ln.get_ydata()
            for xv, yv in zip(np.atleast_1d(xd), np.atleast_1d(yd)):
                if not (np.isfinite(xv) and np.isfinite(yv)):
                    continue
                if (xv < x0 - 1e-9 * span_x or xv > x1 + 1e-9 * span_x
                        or yv < y0 - 1e-9 * span_y or yv > y1 + 1e-9 * span_y):
                    bad.append(f'{ax.get_title() or ax.get_ylabel()} ({xv:.4g},{yv:.4g})')
    return bad


def export(fig, name, rows, header, csv_name=None):
    """Save PNG + PDF at final size, run the QA pass, and write the value CSV.

    Every plotted value is written one per row, so the figure can be audited
    against the analysis output without re-reading the plot.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    png, pdf = f'{FIGDIR}/{name}.png', f'{FIGDIR}/{name}.pdf'
    fig.canvas.draw()
    hits = _text_overlaps(fig)
    out = _artists_outside(fig)
    ends = _endpoints_outside(fig)
    kw = dict(bbox_inches='tight', pad_inches=0.02, facecolor='white')
    fig.savefig(png, dpi=300, **kw)
    fig.savefig(pdf, **kw)
    pts = sorted({t.get_fontsize() for t in fig.findobj(mpl.text.Text)
                  if t.get_text().strip() and t.get_visible()})
    print(f'{name}: text-overlap {len(hits)} | outside-canvas {len(out)} | '
          f'CI/point outside axes {len(ends)} | sizes {pts}')
    if hits:
        print('   overlaps:', hits)
    if out:
        print('   outside:', out[:8])
    if ends:
        print('   endpoints:', ends[:8])
    if rows is not None:
        csv_name = csv_name or f'{name}_values.csv'
        path = os.path.join(CSVDIR, csv_name)
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            fh.write(','.join(header) + '\n')
            for r in rows:
                fh.write(','.join('' if v is None else str(v) for v in r) + '\n')
        print(f'   wrote {path} ({len(rows)} rows)')
    plt.close(fig)
    return dict(overlaps=hits, outside=out, endpoints=ends, png=png, pdf=pdf)
