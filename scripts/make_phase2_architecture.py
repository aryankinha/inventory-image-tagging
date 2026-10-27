"""Draw the Phase 2 network and the hybrid branch into reports/phase2/figures/architecture.png."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(__file__).resolve().parents[1] / 'reports/phase2/figures/architecture.png'
fig, axis = plt.subplots(figsize=(9, 3.3))
axis.set_xlim(0, 22)
axis.set_ylim(0, 7)
axis.axis('off')


def box(x, y, w, h, text, color):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=color, ec='#444', lw=1))
    axis.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8.5, linespacing=1.15)


def arrow(x1, y1, x2, y2):
    axis.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', color='#444', lw=1))


blue, orange, green, gray = '#cfe2f3', '#fce5cd', '#d9ead3', '#eeeeee'
steps = [('Picture\n1x28x28', gray), ('Conv 3x3, 32\nBatchNorm\nReLU, pool\n32x14x14', blue), ('Conv 3x3, 64\nBatchNorm\nReLU, pool\n64x7x7', blue),
         ('Flatten\n3136', gray), ('Dense 128\nReLU\nDropout 0.2', orange), ('Dense 64\nReLU', orange)]
x = 0.2
for index, (text, color) in enumerate(steps):
    box(x, 3.6, 2.4, 2.6, text, color)
    if index:
        arrow(x - 0.4, 4.9, x, 4.9)
    x += 2.8
axis.text(6.2, 6.7, 'Convolution blocks learn local patterns', fontsize=9, ha='center', color='#1c4587')
axis.text(15.4, 6.7, '64 learned numbers per picture', fontsize=9, ha='center', color='#b45f06')
# two ways to finish
box(18.3, 4.6, 3.4, 1.6, 'CNN head\nDropout, Linear 10\nsoftmax', green)
box(18.3, 0.6, 3.4, 1.6, 'Hybrid head\nRandom Forest\n(Phase 1 classifier)', green)
arrow(16.7, 5.0, 18.3, 5.4)
arrow(16.0, 3.6, 18.3, 1.4)
axis.text(8.2, 1.9, 'CNN only: train the whole network.\nHybrid: keep the trained network,\nfit a classical classifier on the 64 numbers.', fontsize=9, va='center')
axis.text(20, 0.15, 'category (10)', fontsize=9, ha='center')
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print('saved', out)
