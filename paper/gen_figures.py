"""
Regenerate all paper figures — black & white, minimal style.
Output: paper/training_loss.pdf, paper/pipeline.pdf, paper/lora_architecture.pdf
"""

import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PAPER_DIR)

BW = {                       # grayscale palette
    'white': '#FFFFFF',
    'light': '#F0F0F0',
    'mid':   '#CCCCCC',
    'dark':  '#888888',
    'black': '#000000',
    'text':  '#000000',
    'dim':   '#555555',
    'line':  '#333333',
}


# ================================================================
# 1. training_loss.pdf
# ================================================================
def gen_training_loss():
    path = os.path.join(ROOT_DIR, 'medical_qwen_lora', 'checkpoint-3750', 'trainer_state.json')
    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    ts, tl = [], []
    es, el = [], []
    for e in state['log_history']:
        if 'loss' in e and 'eval_loss' not in e:
            ts.append(e['step']); tl.append(e['loss'])
        if 'eval_loss' in e:
            es.append(e['step']); el.append(e['eval_loss'])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ts, tl, color='#333333', linewidth=1.0, label='Training Loss')
    ax.plot(es, el, color='#000000', linewidth=1.8,
            marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.2,
            label='Evaluation Loss')

    ax.set_xlabel('Training Step', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training and Evaluation Loss During LoRA Fine-Tuning', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.set_xlim(0, max(ts) + 100)
    ax.set_ylim(1.3, 2.4)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_DIR, 'training_loss.pdf'), format='pdf')
    plt.close(fig)
    print(f'[OK] training_loss.pdf')


# ================================================================
# 2. pipeline.pdf  —  black & white flowchart
# ================================================================
def gen_pipeline():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def box(x, y, w, h, text, subtext=''):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2 + 0.08, text, ha='center', va='center',
                fontsize=10.5, fontweight='bold')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.28, subtext, ha='center', va='center',
                    fontsize=8, color='#555555')

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Main stages
    box(0.3, 3.8, 2.2, 1.3, 'Data Source', 'HuatuoGPT Dataset\n276,042 QA pairs')
    box(2.8, 3.8, 2.2, 1.3, 'Preprocessing',
        'ChatML formatting\nSystem prompt\nTokenization filter\nMax 1,024 tokens')
    box(5.3, 3.8, 2.2, 1.3, 'LoRA Fine-Tuning',
        'Base: Qwen2-1.5B-Instruct\nLoRA: r=16, a=32\nBF16 mixed precision\nRTX 3060 ~5 hours')
    box(7.8, 3.8, 1.9, 1.3, 'MedQwen',
        'Fine-tuned medical\ndialogue model\n1.5B parameters')
    box(3.0, 1.0, 2.4, 1.2, 'Evaluation', 'PPL  |  ROUGE/BLEU\nBERTScore  |  Human Eval')
    box(6.0, 1.0, 2.4, 1.2, 'Deployment', 'Web interface\nInteractive inference\nConsumer GPU')

    # Horizontal arrows
    arr(2.5, 4.45, 2.8, 4.45)
    arr(5.0, 4.45, 5.3, 4.45)
    arr(7.5, 4.45, 7.8, 4.45)

    # MedQwen → Eval & Deploy (T-junction)
    split_y = 2.6
    ecx, dcx = 4.2, 7.2
    ax.plot([8.75, 8.75], [3.8, split_y], color='black', lw=1.5)
    ax.plot([ecx, 8.75], [split_y, split_y], color='black', lw=1.5)
    ax.annotate('', xy=(ecx, 2.2), xytext=(ecx, split_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(dcx, 2.2), xytext=(dcx, split_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.text(5, 0.2, 'All on a single consumer GPU (RTX 3060, 12 GB)',
            ha='center', va='center', fontsize=9, fontstyle='italic', color='#666666')

    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_DIR, 'pipeline.pdf'), format='pdf')
    plt.close(fig)
    print('[OK] pipeline.pdf')


# ================================================================
# 3. lora_architecture.pdf  —  black & white
# ================================================================
def gen_lora_architecture():
    fig = plt.figure(figsize=(10, 5.5))

    # ======== LEFT: Transformer Layer ========
    ax = fig.add_axes([0.02, 0.05, 0.48, 0.90])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(5, 9.55, 'Transformer Layer with LoRA Adapters',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Box outline only — no fill
    def dbox(x, y, w, h, title, sub='', lw=1.5):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           facecolor='white', edgecolor='black', linewidth=lw)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2 + 0.05, title, ha='center', va='center',
                fontsize=9, fontweight='bold')
        if sub:
            ax.text(x + w/2, y + h/2 - 0.25, sub, ha='center', va='center',
                    fontsize=7, color='#444444')

    def anno(y_head, y_tail, x=4.5):
        ax.annotate('', xy=(x, y_head), xytext=(x, y_tail),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    def lora_badge(x, y, w=1.3, h=0.65):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor='black', lw=1.2,
                           linestyle='dashed')
        ax.add_patch(r)
        ax.text(x + w/2, y + h - 0.22, 'LoRA', ha='center', va='center',
                fontsize=7.5, fontweight='bold')
        ax.text(x + w/2, y + 0.18, '(r=16)', ha='center', va='center',
                fontsize=6.5, color='#444444')

    CX = 4.5; BW = 5.5; LX = 7.5

    # 1. Input Embeddings
    dbox(CX - BW/2, 8.3, BW, 0.6, 'Input Embeddings')
    anno(7.63, 8.3)  # arrowhead enters stack at top

    # 2. x24 stack  (dashed border: y=7.03 to 7.63)
    sx, sy = CX - 1.6, 7.05
    for i in range(3):
        r = FancyBboxPatch((sx, sy + i*0.18), 3.2, 0.18,
                           boxstyle="round,pad=0.02", facecolor='white',
                           edgecolor='#888888', lw=0.6)
        ax.add_patch(r)
    r = FancyBboxPatch((sx, 7.03), 3.2, 0.6,
                       boxstyle="round,pad=0.04", facecolor='none',
                       edgecolor='#888888', lw=0.8, linestyle='--')
    ax.add_patch(r)
    ax.text(CX, sy + 0.22, 'x24 Transformer Layers', ha='center', va='center',
            fontsize=8, fontstyle='italic', color='#444444')
    ax.plot([CX, CX], [7.63, 7.03], color='black', lw=1.5)  # through stack interior
    anno(6.35, 7.03)  # arrow exits stack bottom → LN1 top

    # 3. Layer Norm 1  (y=5.85, h=0.5 → 5.85–6.35)
    dbox(CX - 1.8, 5.85, 3.6, 0.5, 'Layer Norm')
    anno(4.9, 5.85)  # arrow enters MHA top

    # 4. MHA  (y=3.7, h=1.2 → 3.7–4.9)
    dbox(CX - BW/2, 3.7, BW, 1.2, 'Multi-Head Self-Attention',
         '(Q, K, V, O projections)')
    lora_badge(LX, 3.95)
    ax.plot([CX + BW/2, LX], [4.35, 4.3], color='#888888', lw=0.8, linestyle='--')
    ax.plot([CX, CX], [4.9, 3.7], color='black', lw=1.5)  # through MHA interior
    anno(3.25, 3.7)  # arrow exits MHA bottom → Add1 top

    # 5. Add & Residual 1  (y=2.7, h=0.55 → 2.7–3.25)
    dbox(CX - 2.0, 2.7, 4.0, 0.55, 'Add & Residual')
    anno(2.15, 2.7)  # arrow enters LN2 top

    # 6. Layer Norm 2  (y=1.65, h=0.5 → 1.65–2.15)
    dbox(CX - 1.8, 1.65, 3.6, 0.5, 'Layer Norm')
    anno(1.0, 1.65)  # arrow enters FFN top

    # 7. FFN  (y=0.1, h=0.9 → 0.1–1.0)
    dbox(CX - BW/2, 0.1, BW, 0.9, 'FFN (Gate, Up, Down)')
    lora_badge(LX, 0.25)
    ax.plot([CX + BW/2, LX], [0.55, 0.55], color='#888888', lw=0.8, linestyle='--')
    ax.plot([CX, CX], [1.0, 0.1], color='black', lw=1.5)  # through FFN interior
    anno(-0.3, 0.1)  # arrow exits FFN bottom → Add2 top

    # 8. Add & Residual 2
    dbox(CX - 2.0, -0.85, 4.0, 0.55, 'Add & Residual')

    ax.text(CX, -1.2, 'Output to Next Layer', ha='center', va='center',
            fontsize=8.5, fontstyle='italic', color='#666666')

    # Legend (black/white)
    for lx, lt in [(0.5, 'Module'), (3.0, 'LoRA'), (5.5, 'Norm'), (7.5, 'Add')]:
        r = FancyBboxPatch((lx, -1.7), 0.6, 0.22, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='black', lw=0.5)
        ax.add_patch(r)
        ax.text(lx + 0.3, -1.59, lt, ha='center', va='center', fontsize=6.5)

    # ======== RIGHT: LoRA Low-Rank Adaptation ========
    ax = fig.add_axes([0.52, 0.12, 0.45, 0.78])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.text(5, 9.55, 'LoRA Low-Rank Adaptation',
            ha='center', va='center', fontsize=12, fontweight='bold')

    def rbox(x, y, w, h, txt, sub='', lw=1.5, ls='solid', fs=9):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor='white', edgecolor='black', linewidth=lw, linestyle=ls)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2 + 0.06, txt, ha='center', va='center',
                fontsize=fs, fontweight='bold')
        if sub:
            ax.text(x + w/2, y + h/2 - 0.28, sub, ha='center', va='center',
                    fontsize=7, color='#444444')

    # Input x
    rbox(4.2, 8.2, 1.6, 0.6, '$x$', fs=12)

    # Fork arrows
    ax.annotate('', xy=(3.5, 7.5), xytext=(5, 8.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
    ax.annotate('', xy=(6.5, 7.5), xytext=(5, 8.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    # W0 (frozen) — solid outline
    fs = 9
    rbox(0.8, 5.8, 4.0, 1.4, '$W_0$', 'Frozen pre-trained\nweight matrix', lw=1.8)
    ax.text(2.8, 6.5, r'$\in\mathbb{R}^{d\times k}$', ha='center', va='center',
            fontsize=8.5, color='#444444')

    # B — dashed outline (trainable)
    rbox(6.0, 6.6, 3.2, 0.8, '$B$', r'$\in\mathbb{R}^{d\times r}$ (Trainable)',
         lw=1.5, ls='dashed')
    rbox(6.0, 5.2, 3.2, 0.8, '$A$', r'$\in\mathbb{R}^{r\times k}$ (Trainable)',
         lw=1.5, ls='dashed')
    ax.text(7.6, 6.0, r'$r\ll\min(d,k)$', ha='center', va='center',
            fontsize=8, fontstyle='italic', color='#444444')
    ax.text(7.6, 4.7, r'$\rightarrow$ $\Delta W=BA$', ha='center', va='center',
            fontsize=9, fontweight='bold')

    # Down arrows
    ax.annotate('', xy=(2.8, 5.0), xytext=(2.8, 5.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
    ax.annotate('', xy=(7.6, 4.2), xytext=(7.6, 5.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))

    # Summation circle
    circle = plt.Circle((2.8, 3.8), 0.5, facecolor='white', edgecolor='black', lw=1.8)
    ax.add_patch(circle)
    ax.text(2.8, 3.8, '+', ha='center', va='center', fontsize=14, fontweight='bold')

    # BA → summer (curved)
    ax.annotate('', xy=(2.8, 4.3), xytext=(7.6, 4.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, linestyle='dashed',
                                connectionstyle='arc3,rad=0.3'))

    # Summer → output
    ax.annotate('', xy=(2.8, 3.0), xytext=(2.8, 3.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Output
    rbox(0.8, 1.5, 4.0, 1.2, r'$h=W_0x+\Delta Wx$',
         r'$=W_0x+BAx$', lw=2)
    ax.text(8.5, 2.1, 'Standard\npath', ha='center', va='center', fontsize=7)
    ax.text(8.5, 1.3, 'LoRA\npath', ha='center', va='center',
            fontsize=7, fontweight='bold')
    ax.annotate('', xy=(4.8, 2.1), xytext=(2.8, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
    ax.annotate('', xy=(7.6, 2.1), xytext=(7.6, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8, linestyle='dashed'))

    ax.text(5, 0.5, 'Only $A$ and $B$ are updated during fine-tuning;',
            ha='center', va='center', fontsize=8, fontstyle='italic', color='#555555')
    ax.text(5, 0.15, '$W_0$ remains frozen.',
            ha='center', va='center', fontsize=8, fontstyle='italic', color='#555555')

    fig.savefig(os.path.join(PAPER_DIR, 'lora_architecture.pdf'), format='pdf')
    plt.close(fig)
    print('[OK] lora_architecture.pdf')


# ================================================================
if __name__ == '__main__':
    gen_training_loss()
    gen_pipeline()
    gen_lora_architecture()
    print('\nAll figures regenerated (black & white).')
