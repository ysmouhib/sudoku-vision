"""Shared plot style: one fixed color per model, recessive axes and grid."""
import matplotlib

matplotlib.use("Agg")  # write files only, never open windows
import matplotlib.pyplot as plt  # noqa: E402

MODEL_COLORS = {"linear": "#2a78d6", "mlp": "#eb6834", "cnn": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
GOOD, BAD = "#0ca30c", "#d03b3b"


def style(ax: plt.Axes) -> None:
    """Make the grid and axes recessive and the text dark."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def save(fig: plt.Figure, path: str) -> None:
    """Save a figure at a fixed resolution and close it."""
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
