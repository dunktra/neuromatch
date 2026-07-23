"""Generate individual component visuals for Slide 1."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Component 1: Task schematic
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

# Random dot kinematogram
ax.add_patch(plt.Circle((1.2, 1.5), 0.7, fill=False, edgecolor="gray", linewidth=1.5))
rng = np.random.default_rng(42)
for _ in range(40):
    rx = 0.6 + rng.random() * 1.2
    ry = 0.9 + rng.random() * 1.2
    if (rx - 1.2) ** 2 + (ry - 1.5) ** 2 < 0.45:
        ax.plot(rx, ry, ".", color="gray", markersize=4)
# Motion direction arrow inside
ax.annotate("", xy=(1.7, 2.0), xytext=(0.9, 1.2),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=2.5))
ax.text(1.2, 0.35, "Random-dot\nmotion stimulus", ha="center", va="top", fontsize=10)

# Arrow to dial
ax.annotate("", xy=(3.8, 1.5), xytext=(2.2, 1.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
ax.text(3.0, 1.8, "estimate\ndirection", ha="center", fontsize=9, color="#555555")

# Circular dial
ax.add_patch(plt.Circle((4.5, 1.5), 0.6, fill=False, edgecolor="black", linewidth=2))
# Dial ticks
for angle in range(0, 360, 30):
    a = np.radians(angle)
    x1 = 4.5 + 0.55 * np.cos(a)
    y1 = 1.5 + 0.55 * np.sin(a)
    x2 = 4.5 + 0.6 * np.cos(a)
    y2 = 1.5 + 0.6 * np.sin(a)
    ax.plot([x1, x2], [y1, y2], "k-", linewidth=0.8)
# Pointer
ax.plot([4.5, 4.95], [1.5, 1.95], "k-", linewidth=2.5)
ax.plot(4.5, 1.5, "ko", markersize=5)
ax.text(4.5, 0.35, "Response dial", ha="center", va="top", fontsize=10)

# Arrow to conditions
ax.annotate("", xy=(7.0, 1.5), xytext=(5.3, 1.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.5))

# Conditions box
cond_text = ("Conditions\n\n"
             "4 prior widths:  10° / 20° / 40° / 80°\n"
             "3 coherences:  0.06 / 0.12 / 0.24\n"
             "Prior mean:  225° (fixed)\n\n"
             "12 subjects  ·  83,210 trials")
ax.text(8.3, 1.5, cond_text, ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="gray", linewidth=1.5))

fig.savefig(os.path.join(OUT_DIR, "slide1_task.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Component 1 (task) saved")


# ═══════════════════════════════════════════════════════════════════════
# Component 2: Bayesian Blending model
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-4, 4)
ax.set_ylim(-0.8, 4)
ax.axis("off")
ax.set_title("Bayesian Blending", fontsize=16, fontweight="bold", color="#333333", pad=15)

x = np.linspace(-3.5, 3.5, 300)

# Measurement
y_meas = np.exp(-0.5 * (x - 0.3) ** 2 / 0.7) * 1.8
ax.fill_between(x, 0, y_meas, alpha=0.2, color="steelblue")
ax.plot(x, y_meas, color="steelblue", linewidth=2, label="measurement $m$ ~ VM(θ, κ$_{sens}$)")

# Prior
y_prior = np.exp(-0.5 * (x - 1.8) ** 2 / 0.5) * 2.2
ax.fill_between(x, 0, y_prior, alpha=0.15, color="indianred")
ax.plot(x, y_prior, color="indianred", linewidth=2, linestyle="--", label="prior VM(μ$_p$, κ$_p$)")

# Combined posterior
y_post = np.exp(-0.5 * (x - 0.9) ** 2 / 0.35) * 3.0
ax.fill_between(x, 0, y_post, alpha=0.25, color="purple")
ax.plot(x, y_post, color="purple", linewidth=3, label="posterior (blend)")

# Report arrow
ax.annotate("", xy=(0.9, 3.3), xytext=(0.9, 3.8),
            arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
ax.text(0.9, 3.9, "single report", ha="center", fontsize=11, color="purple", fontweight="bold")

# Formula
ax.text(-3.5, 2.5, r"$R e^{i\mu_R} = \kappa_{sens} e^{im} + \kappa_{prior} e^{i\mu_{prior}}$",
        fontsize=12, color="#333333")

ax.text(0, -0.6, "Always unimodal — combines evidence + prior into one posterior",
        ha="center", fontsize=11, style="italic", color="#555555")
ax.legend(fontsize=10, loc="upper left")

fig.savefig(os.path.join(OUT_DIR, "slide1_bayes.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Component 2 (Bayes) saved")


# ═══════════════════════════════════════════════════════════════════════
# Component 3: Switching Observer model
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-4, 4)
ax.set_ylim(-0.8, 4.5)
ax.axis("off")
ax.set_title("Switching Observer (Laquitaine & Gardner, 2018)", fontsize=16,
             fontweight="bold", color="#333333", pad=15)

x = np.linspace(-3.5, 3.5, 300)

# Measurement
y_meas = np.exp(-0.5 * (x - 0.3) ** 2 / 0.7) * 1.5
ax.fill_between(x, 0, y_meas, alpha=0.15, color="steelblue")
ax.plot(x, y_meas, color="steelblue", linewidth=1.5, label="measurement $m$")

# Prior
y_prior = np.exp(-0.5 * (x - 1.8) ** 2 / 0.5) * 1.8
ax.fill_between(x, 0, y_prior, alpha=0.12, color="indianred")
ax.plot(x, y_prior, color="indianred", linewidth=1.5, linestyle="--", label="prior")

# Two outcome peaks
y_peak1 = np.exp(-0.5 * (x - 0.3) ** 2 / 0.6) * 2.5
y_peak2 = np.exp(-0.5 * (x - 1.8) ** 2 / 0.4) * 2.3
ax.fill_between(x, 0, y_peak1, alpha=0.2, color="steelblue")
ax.plot(x, y_peak1, color="steelblue", linewidth=2.5)
ax.fill_between(x, 0, y_peak2, alpha=0.2, color="indianred")
ax.plot(x, y_peak2, color="indianred", linewidth=2.5)

# Context posterior box
ax.text(-3.2, 3.8, "$P(H_{vm}|m)$", fontsize=12, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", linewidth=1.5))

# Arrows from context posterior to each peak
ax.annotate("", xy=(0.3, 2.8), xytext=(-2.5, 3.6),
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=2, connectionstyle="arc3,rad=0.2"))
ax.annotate("", xy=(1.8, 2.6), xytext=(-2.5, 3.6),
            arrowprops=dict(arrowstyle="->", color="indianred", lw=2, connectionstyle="arc3,rad=-0.2"))

ax.text(0.3, 3.0, "report\nmeasurement", ha="center", fontsize=9, color="steelblue", fontweight="bold")
ax.text(1.8, 2.8, "report\nprior", ha="center", fontsize=9, color="indianred", fontweight="bold")

# p_uniform knob
ax.text(-3.2, 2.2, "$p_{uniform}$\narbitration\nknob", fontsize=10, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e8", edgecolor="gray", linewidth=1.5))

ax.text(0, -0.6, "Can be bimodal — commits to one source per trial, never blends",
        ha="center", fontsize=11, style="italic", color="#555555")
ax.legend(fontsize=9, loc="upper right")

fig.savefig(os.path.join(OUT_DIR, "slide1_switching.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Component 3 (switching) saved")


# ═══════════════════════════════════════════════════════════════════════
# Component 4: Key difference callout
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.5)
ax.axis("off")

# Left side: blending
ax.add_patch(FancyBboxPatch((0.3, 0.3), 4, 1.9, boxstyle="round,pad=0.2",
                             facecolor="#e3f2fd", edgecolor="#2196F3", linewidth=1.5))
ax.text(2.3, 1.65, "Blending", fontsize=13, fontweight="bold", ha="center", color="#1565C0")
ax.text(2.3, 1.0, "measurement + prior → one posterior\n→ always unimodal", fontsize=10,
        ha="center", va="center", color="#333333")

# VS
ax.text(5, 1.25, "vs.", fontsize=14, ha="center", va="center", color="gray", fontweight="bold")

# Right side: switching
ax.add_patch(FancyBboxPatch((5.7, 0.3), 4, 1.9, boxstyle="round,pad=0.2",
                             facecolor="#fce4ec", edgecolor="#E91E63", linewidth=1.5))
ax.text(7.7, 1.65, "Switching", fontsize=13, fontweight="bold", ha="center", color="#C62828")
ax.text(7.7, 1.0, "measurement OR prior → pick one source\n→ can be bimodal across trials", fontsize=10,
        ha="center", va="center", color="#333333")

fig.savefig(os.path.join(OUT_DIR, "slide1_keydiff.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Component 4 (key difference) saved")

print(f"\nAll components saved to {OUT_DIR}/")
