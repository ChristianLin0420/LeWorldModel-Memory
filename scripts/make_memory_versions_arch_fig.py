#!/usr/bin/env python3
"""Generate the consolidated architecture/ablation map for the learnable-memory study."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'figures' / 'fig_memory_versions_arch.png'


def card(ax, x, y, w, h, title, body, color, *, title_size=12, body_size=8.5,
         edge='#263238', linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.04',
        facecolor=color, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    title_color = '#f7f9fb' if luminance < 0.43 else '#17202a'
    body_color = '#f0f3f5' if luminance < 0.43 else '#263238'
    ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color=title_color)
    ax.text(x + w / 2, y + h / 2 - 0.12, body, ha='center', va='center',
            fontsize=body_size, color=body_color, linespacing=1.35)


def arrow(ax, start, end, label=''):
    patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                            color='#546e7a', linewidth=1.7)
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, start[1] + 0.18, label,
                ha='center', va='bottom', fontsize=8.2, color='#546e7a')


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 44.8), dpi=180)
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 44.8)
    ax.axis('off')

    ax.text(9, 44.45, 'Learnable-memory architecture map: V1–V13 and tested/adaptive controls',
            ha='center', fontsize=18, fontweight='bold', color='#17202a')
    ax.text(9, 44.05,
            'Architecture-changing variants are shown explicitly; seeds, optimizer settings, '
            'mask shifts, and K/M/τ sweeps are experimental settings.',
            ha='center', fontsize=10.5, color='#607d8b')

    xs = tuple(0.04 + index * 1.38 for index in range(13))
    width, top_y, top_h = 1.14, 41.25, 2.05
    card(ax, xs[0], top_y, width, top_h, 'SMT-v1',
         r'value-gated EMA write' + '\n' + r'$i_t\odot z_t$; old state still decays' + '\n'
         + 'softmax horizon read', '#e3f2fd', body_size=4.75)
    card(ax, xs[1], top_y, width, top_h, 'SMT-v2',
         'same erasing write\nindependent sigmoid reads\n'
         'larger mass; gates become static', '#dcedc8', body_size=4.75)
    card(ax, xs[2], top_y, width, top_h, 'SMT-v3-W',
         'whole-update scalar gate\n$g_t=0$ exactly freezes all EMA banks\n'
         'global simplex; action-blind', '#fff3e0', body_size=4.75)
    card(ax, xs[3], top_y, width, top_h, 'HACSM-v4',
         'three belief levels $\\tau=\\{2,8,32\\}$\naction prior $p_t=T(m_{t-1},a_{t-1})$\n'
         'selective correction + fixed auxiliary', '#e1bee7', body_size=4.45)
    card(ax, xs[4], top_y, width, top_h, 'HACSSM-v5',
         'two fast/medium states\nhard-monotone channel gains\n'
         'action predict/correct\nboundary-only shaping', '#d1c4e9', body_size=4.45)
    card(ax, xs[5], top_y, width, top_h, 'HACSSM-v6',
         'fixed scalar $\\tau=\\{2,8\\}$ anchor\ndense visible-endpoint\n'
         'same-level action consistency', '#c5cae9', body_size=4.55)
    card(ax, xs[6], top_y, width, top_h, 'HACSSM-v7',
         'level-specific action heads\nstatic/dynamic gate shrinkage\n'
         'EMA counterfactual recovery', '#b39ddb', body_size=4.45)
    card(ax, xs[7], top_y, width, top_h, 'SAS-PC-v8',
         'one physical shared action head\nlearned gate shrinkage\n'
         'joint read; no internal auxiliary', '#9575cd', body_size=4.45)
    card(ax, xs[8], top_y, width, top_h, 'LOIF-v9',
         'learned ordered poles\nevidence scale + gain\n'
         'inverse-scale fusion\ncomplete; locked NO_GO', '#7e57c2', body_size=4.20)
    card(ax, xs[9], top_y, width, top_h, 'ORBIT-v10-J',
         'one persistent $D$-state\n2 action Givens layers\nno decay / no horizon\n'
         'joint VICReg host\nNO LAUNCH', '#673ab7', body_size=4.05)
    card(ax, xs[10], top_y, width, top_h, 'KDIO-v11',
         'configuration + velocity\n$\\gamma\\,\\mathrm{qf}(M)$ action lift\nkick-drift integration\n'
         'live suffix + detached rank\nNO_GO; official 0/400',
         '#512da8', body_size=2.85)
    card(ax, xs[11], top_y, width, top_h, 'SIRO-v12',
         'anchor + residual + action\nstable FWL fit; paired OAS K\n'
         '28/28 artifacts; rank 21/28\nNO_GO; no 100e',
         '#3949ab', body_size=2.65)
    card(ax, xs[12], top_y, width, top_h, 'CF-HIRO',
         'v13 split-agreement Hankel\nnormal modal state + complement\nfixed DARE gain\nREADY_NOT_RUN 0/36',
         '#283593', title_size=9.0, body_size=2.25)
    for index in range(12):
        arrow(ax, (xs[index] + width, 42.27), (xs[index + 1], 42.27))

    ax.text(0.35, 40.83,
            'V8 shared-action shrinkage controls (34,566 compact / 36,102 expanded parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v8_controls = (
        ('compact', 'physical shared head\nlearned shrinkage\njoint read', '#9575cd'),
        ('dynamic', '$\\rho_f=\\rho_m=1$\nretrained dynamic endpoint', '#f8bbd0'),
        ('static', '$\\rho_f=\\rho_m=0$\nretrained static endpoint', '#fff3e0'),
        ('level-action', 'separate fast/medium heads\naction-tying control', '#d1c4e9'),
        ('redundant', 'equal head halves; averaged\nstatistical equivalence', '#ffecb3'),
        ('no-action', 'shared action contribution 0\nstructural receipt', '#ffe0b2'),
        ('single', 'medium-only read\nboth states retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v8_controls):
        card(ax, 0.15 + index * 2.55, 39.15, 2.30, 1.34, title, body, color,
             title_size=8.5, body_size=6.2)

    ax.text(0.35, 38.75,
            'V9 LOIF completed controls (34,563 memory parameters; 325 cells; paired full-vs-control delta)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v9_controls = (
        ('loifv9', '+2.551% vs SSM\nrank 7/13; NO_GO', '#7e57c2'),
        ('fixed-$\\alpha$', '$\\alpha=(e^{-1/2},e^{-1/8})$\nfull delta = -0.671%', '#fff3e0'),
        ('global-$R$', '$R_t=softplus(b_R)+\\epsilon$\nfull delta = +30.556%', '#f8bbd0'),
        ('innovation-only', 'disconnect direct latent\nfull delta = +19.189%', '#e1f5fe'),
        ('latent-only', 'disconnect innovation\nfull delta = +0.475%', '#fce4ec'),
        ('uniform-fusion', 'uniform prior + output\nfull delta = +0.340%', '#b3e5fc'),
        ('no-action', 'zero action innovation\nfull delta = +8.293%', '#ffe0b2'),
        ('single-bank', 'fast state disconnected\nfull delta = -2.220%', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v9_controls):
        card(ax, 0.12 + index * 2.22, 37.07, 2.05, 1.34, title, body, color,
             title_size=7.25, body_size=5.15)

    ax.text(0.35, 36.67,
            'V10-J ORBIT modes (34,562 parameters at D=128/A=6; full host audit only, official 0/225)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v10_controls = (
        ('orbitv10', 'normalized 2-layer rotations\n5-task 100e host audit\nNO LAUNCH', '#673ab7'),
        ('no-action', '$T(a)=I$\ntests action transport\nPENDING', '#ffe0b2'),
        ('additive', 'V8-like additive prior\nsame tensor schema\nPENDING', '#ffecb3'),
        ('scaled', 'unnormalized complex blocks\ntests exact isometry\nPENDING', '#f8bbd0'),
        ('static', '$g_t=\\sigma(b)$\ntests innovation evidence\nPENDING', '#fff3e0'),
    )
    for index, (title, body, color) in enumerate(v10_controls):
        card(ax, 0.20 + index * 3.55, 34.97, 3.22, 1.34, title, body, color,
             title_size=8.5, body_size=5.8)

    ax.text(0.35, 34.53,
            'V11 registered grid: 16 × 5 × 5 = 400 cells / 40,000 epoch rows — UNLAUNCHED 0/400 after development NO_GO; '
            '17,796 nominal optimizer + 8,255 closed-form OAS = 26,051 total scalars',
            fontsize=10.4, fontweight='bold', color='#37474f')
    v11_controls = (
        ('SSM', 'baseline\none $D$-state', '#e3f2fd'),
        ('compact V8', 'baseline\nadditive two-state', '#e3f2fd'),
        ('ORBIT V10', 'baseline\northogonal prior', '#e3f2fd'),
        ('full KDIO', '$\\gamma=e^{\\log\\gamma}>0$\n$U=\\mathrm{qf}(M), U^\\top U=I_A$\nlive suffix + detached rank', '#512da8'),
        ('fixedscale', '$\\gamma=1$ exactly\n$\\log\\gamma$ tensor retained\n17,795 active scalars', '#fff3e0'),
        ('free geometry', '$\\gamma\\sqrt{A}M/\\|M\\|_F$\nQR bypassed; scale identifiable', '#fff3e0'),
        ('nocalibration', '$\\mu=0,C=I$\nfit disabled', '#fff3e0'),
        ('diagonal OAS', 'epoch OAS fit\ndiagonal only', '#fff3e0'),
        ('h1', 'live suffix + rank\n$k=1$ only', '#e8f5e9'),
        ('firstorder', 'architecture control\nno velocity carry', '#fff3e0'),
        ('nodrift', 'architecture control\n$q^-=q$', '#fff3e0'),
        ('noautonomy', 'architecture control\n$w_q=b_f=0$', '#fff3e0'),
        ('noaction', 'architecture control\nzero action input', '#fff3e0'),
        ('noactionswap', 'rank diagnosed; optimized term 0\nlive suffix retained', '#e8f5e9'),
        ('nosuffix', 'live suffix + rank off\nfull inference retained', '#e8f5e9'),
        ('noreliability', 'OAS stats retained\n$r=1$', '#fff3e0'),
    )
    for index, (title, body, color) in enumerate(v11_controls):
        row, col = divmod(index, 4)
        card(ax, 0.20 + col * 4.43, 32.85 - row * 1.615, 4.02, 1.34,
             title, body, color, title_size=8.2, body_size=5.55)

    ax.text(9, 27.62,
            'V11 closeout: rawdiff .577987 (best), default .581525, legal integrator .476157; '
            'all V11b late changes negative → NO_GO / NO_LAUNCH; official matrix stays 0/400.',
            ha='center', fontsize=8.25, fontweight='bold', color='#b71c1c')

    ax.text(0.35, 27.15,
            'V12 SIRO completed adaptive screen: seven designs × four tasks = 28/28 artifact-valid cells; 21 rank-valid / 7 rank-fail',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v12_controls = (
        ('full centered', '$c=z_0$; $h=c+r+u$\nNMSE .935218 • rank 3/4', '#a5d6a7'),
        ('spectral shrink', '$h=c+r+Ru$; shared-$A$ parity $R$\nNMSE .958373 • rank 3/4', '#e1bee7'),
        ('identity $A$', '$A=I$ in $r,u$\nNMSE 1.486104 • rank 3/4', '#ffe0b2'),
        ('identity $K$', '$K=I$\nNMSE 1.351535 • rank 3/4', '#fff3e0'),
        ('no action', '$u\\equiv0$\nNMSE 1.003325 • rank 3/4', '#ffccbc'),
        ('absolute no-anchor', '$c=0,r_0=z_0$; fit absolute $z$\nNMSE 1.495009 • rank 2/4', '#b3e5fc'),
        ('retrained V11', 'rawdiff KDIO predecessor\nNMSE .564069 • artifacts 4/4', '#d1c4e9'),
    )
    for index, (title, body, color) in enumerate(v12_controls):
        card(ax, 0.15 + index * 2.55, 25.43, 2.30, 1.34, title, body, color,
             title_size=7.8, body_size=5.65)
    card(ax, 0.35, 23.75, 17.30, 1.45,
         'SIRO-v12 closeout: SCREEN_NO_GO / NO 100-EPOCH LAUNCH',
         'Full .935218 vs retrained V11 .564069 vs legal integrator .469803. Shared $A$ + full $R=I$ gives exact '
         '$v=r+u$: $v^-=Av+Ba+b$, so the full split is bookkeeping—not a functionally distinct hierarchy.\n'
         'V12b zero-step replay tested old-history $K$ .723578, identity $K$ .727920, deployed-history LMMSE '
         '$2.008\\times10^{12}$, current-$A$ Riccati .724090, and normal-$A$ Riccati .376639; action partial-$R^2$ positive 0/4 → STOP.',
         '#ffebee', title_size=9.4, body_size=5.25, edge='#b71c1c', linewidth=1.6)

    ax.text(0.35, 23.30,
            'V13 CF-HIRO prospective screen: six same-schema modes + SSM / compact V8 / KDIO; READY_NOT_RUN, 0/36',
            fontsize=10.8, fontweight='bold', color='#37474f')
    v13_controls = (
        ('full', r'complement $c_\perp$' '\npositive-part fold $g$\n' r'normal $A$ + fixed $K_\infty$', '#a5d6a7'),
        ('fullanchor', r'$c=z_0-\mu_z,x_0=0$' '\nfull frozen anchor\nscreen-only', '#b3e5fc'),
        ('triangular', 'retain Schur\ninter-block coupling\nscreen-only', '#f8bbd0'),
        ('noshrink', '$g_i=1$ exactly\nno fold reliability\nretained if launch', '#e1bee7'),
        ('noaction', '$B_{eff}=0$ exactly\nfitted $B$ retained\nretained if launch', '#ffccbc'),
        ('nocorrect', '$K_{eff}=0$ exactly\n' r'fitted $K_\infty$ retained' '\nscreen-only', '#ffe0b2'),
    )
    for index, (title, body, color) in enumerate(v13_controls):
        card(ax, 0.20 + index * 2.95, 21.58, 2.70, 1.30, title, body, color,
             title_size=7.6, body_size=5.35)
    card(ax, 0.35, 19.92, 17.30, 1.36,
         'CF-HIRO-v13 frozen protocol: 36-cell 30e screen; conditional 72-cell 100e wave only after every gate',
         'All-lag split-fold moments → positive-part agreement → full-order Ho–Kalman state → real-normal blocks → '
         'all-lag $B$ refit → paired OAS/DARE. Direct-sum read ' r'$\mu_z+c_\perp+Cx$' '; no online covariance or memory loss.\n'
         'Gates: full rank 4/4; all six numerical; ≥5% vs noaction/external/integrator, ≥2% vs other controls, each 3/4 wins; both RMS >1e-8. '
         'No V13 W&B/result; not cross-fitted, hyperparameter-free, automatically sized, fundamentally novel, or ICLR-ready.',
         '#e8eaf6', title_size=8.7, body_size=5.35, edge='#c62828', linewidth=1.5)

    ax.text(0.35, 19.38, 'Pre-V4 architecture controls used in the mechanism studies',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v3_controls = (
        ('scaled-softmax SMT', 'V1 softmax route × K/2\nmatched initial read mass\nisolates V1→V2 amplitude', '#e1f5fe'),
        ('V3 static', '$g_t=\\sigma(b)$\nremoves input conditioning\nmatched dynamic-gate parameters', '#ffe0b2'),
        ('V3 old-erasing', 'dynamic scalar gate on value only\nold state still decays\nisolates true-freeze semantics', '#ffecb3'),
        ('V3 hard visibility', 'known black/visible mask\nforces freeze/update timing\nnominal parameter control', '#f3e5f5'),
    )
    for index, (title, body, color) in enumerate(v3_controls):
        card(ax, 0.35 + index * 4.42, 17.60, 4.02, 1.42, title, body, color,
             title_size=10.2, body_size=7.6)

    ax.text(0.35, 17.22, 'V4 controls and the slow-removal bridge into V5/V6',
            fontsize=11.5, fontweight='bold', color='#37474f')
    controls = (
        ('full', 'dynamic correction\n+ action prior\n+ hierarchical auxiliary', '#d1c4e9'),
        ('static', '$g_t^k=\\sigma(b_k)$\nremoves input-conditioned\ncorrection timing', '#ffe0b2'),
        ('no-action', '$W_a$ instantiated\naction contribution forced 0\nisolates action transition', '#ffecb3'),
        ('no-aux', 'full online architecture\nauxiliary weight forced 0\nisolates self-supervision', '#c8e6c9'),
        ('single-level', 'dynamic/action states retained\nread fixed to middle $\\tau=8$\nisolates hierarchical read', '#b2ebf2'),
        ('two-level no-aux', '$\\tau=\\{2,8\\}$ only\nfixed scalar gains\nisolates slow removal', '#e0f2f1'),
    )
    for index, (title, body, color) in enumerate(controls):
        card(ax, 0.20 + index * 2.97, 15.42, 2.70, 1.50, title, body, color,
             title_size=9.3, body_size=6.8)

    ax.text(0.35, 15.03, 'V5 parameter-matched mechanism variants (34,820 memory parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v5_controls = (
        ('full', 'learned gains + action\ndynamic correction\nboundary shaping', '#d1c4e9'),
        ('no-aux', 'same online memory\nweight forced 0\nisolates objective', '#c8e6c9'),
        ('fixed-β no-aux', 'initial channel bands frozen\nno boundary shaping\nisolates gain learning', '#e3f2fd'),
        ('no-action', 'learned gains retained\naction contribution 0\naux schedule retained', '#ffecb3'),
        ('static', 'input-independent gates\naction + learned gains\naux schedule retained', '#ffe0b2'),
        ('single-medium', 'both states retained\nread fixed to medium\nisolates joint read', '#b2ebf2'),
        ('band-SSM', '$g=1$, action 0\nlearned two-state spectrum\nstate-matched control', '#eceff1'),
    )
    for index, (title, body, color) in enumerate(v5_controls):
        card(ax, 0.15 + index * 2.55, 13.18, 2.30, 1.48, title, body, color,
             title_size=8.4, body_size=6.2)

    ax.text(0.35, 12.78,
            'V6 fixed-rate inference anchor + complete training-objective/mechanism map '
            '(34,564 memory parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v6_controls = (
        ('full', 'detached source/target\nfast h=1,2; medium h=4,8', '#d1c4e9'),
        ('no-aux', 'same inference\nauxiliary weight = 0', '#c8e6c9'),
        ('aux-no-action', 'zero action only\nin auxiliary rollout', '#ffecb3'),
        ('uniform', 'both levels use\nh={1,2,4,8}', '#e1f5fe'),
        ('source-grad', 'source not detached\ntarget remains detached', '#f8bbd0'),
        ('fast-only', 'only fast auxiliary\nh={1,2}', '#dcedc8'),
        ('medium-only', 'only medium auxiliary\nh={4,8}', '#b2dfdb'),
        ('no-action', 'actions zeroed in\ninference + auxiliary', '#ffe0b2'),
        ('static', 'input-independent\ncorrection gates', '#fff3e0'),
        ('single', 'medium-only read\nboth states/aux retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v6_controls):
        row, col = divmod(index, 5)
        card(ax, 0.20 + col * 3.55, 11.00 - row * 1.72, 3.22, 1.42,
             title, body, color, title_size=9.0, body_size=6.7)

    ax.text(0.35, 8.88,
            'V7 counterfactual-recovery inference/objective map '
            '(36,102 trainable memory parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v7_controls = (
        ('full', 'level-specific action\nlearned gate shrinkage\nbridge + recovery', '#b39ddb'),
        ('no-aux', 'same online inference\nauxiliary weight = 0', '#c8e6c9'),
        ('shared-action', 'average both projected\naction heads per step', '#ffecb3'),
        ('no-shrink', '$\\rho_f=\\rho_m=1$\ndynamic correction only', '#f8bbd0'),
        ('action-only', 'V6-style action rollout\nno recovery correction', '#ffe0b2'),
        ('uniform', 'both levels use\nh={1,2,4,8}', '#e1f5fe'),
        ('no-recovery', 'counterfactual bridge only\nremove restored-frame loss', '#f3e5f5'),
        ('no-action', 'actions zeroed in\ninference + auxiliary', '#fff3e0'),
        ('single', 'medium-only read\nboth states/objectives retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v7_controls):
        row, col = divmod(index, 5)
        card(ax, 0.20 + col * 3.55, 7.10 - row * 1.72, 3.22, 1.42,
             title, body, color, title_size=9.0, body_size=6.7)

    ax.text(0.35, 4.98, 'External controls and historical memory families', fontsize=11.5,
            fontweight='bold', color='#37474f')
    baselines = (
        ('none', 'short predictor only\nno long-range channel'),
        ('fixed EMA', 'short / long / both\nhand-set recurrence'),
        ('fixed multi', '$K=6$ EMA banks\nseparate readouts'),
        ('GRU', 'learned recurrent state\nzero-init residual read'),
        ('diagonal SSM', 'learned channel decay\none $D$-state'),
        ('retrieval', 'causal soft attention\nover past latents'),
        ('OC-SMT', '$M=28$ EMA basis\nhard-concrete reads'),
    )
    for index, (title, body) in enumerate(baselines):
        card(ax, 0.15 + index * 2.55, 3.22, 2.30, 1.42, title, body, '#eceff1',
             title_size=9.0, body_size=6.8, edge='#607d8b', linewidth=1.2)

    card(ax, 0.35, 1.53, 17.30, 1.12,
         'Historical V5–V9 leakage-safe contract; V10–V13 use separate causal-host protocols',
         'fixed DINOv2-PCA targets • $a_t:z_t\\rightarrow z_{t+1}$ • output norm = none '
         '(no cross-window statistics) • blackout targets excluded • '
         'cohort-frozen objective: V5–V8 first-post weight .5; V9 weight 0 • '
         'SIGReg constant/no-gradient • '
         'final epoch, no best-checkpoint selection\n'
         'online W&B for every cell: 200 epoch logs + fixed evaluation-rollout '
         'trace / paired video / hashed artifact',
         '#e8eaf6', title_size=10.5, body_size=7.2, linewidth=1.4)

    ax.text(9, 1.05,
            'V5 complete: 300 runs.  V6 complete: 325 runs.  '
            'V7 complete: 325 runs.  V8 complete: 325 runs, locked negative label.  '
            'V9 complete: 325 runs, locked negative label.  V10-J audit: 5×100 epochs; official 0/225.  '
            'V11 development: 64 cells, NO_GO; official 0/400.  V12 adaptive screen: 28/28 artifacts, NO_GO; 100e 0/28.  '
            'V13: READY_NOT_RUN, screen 0/36; conditional 0/72.',
            ha='center', fontsize=7.9, color='#607d8b')
    ax.text(9, 0.66,
            'Historical V1–V4 cohorts retain their documented protocols; architecture cards do '
            'not imply that raw MSE values are pooled across incompatible target spaces.',
            ha='center', fontsize=8.8, color='#607d8b')
    ax.text(9, 0.27,
            'V11 action/objective variants all fail the development gate; its official matrix was never launched.  '
            'V12 loses to retrained V11 and the legal integrator; its shared-A full read collapses exactly to one $r+u$ state, '
            'and the V12b normal–Riccati repair has no replicated action signal (0/4 tasks).  '
            'V13 architecture/smoke readiness is not performance or novelty evidence.',
            ha='center', fontsize=7.8, fontweight='bold', color='#455a64')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
