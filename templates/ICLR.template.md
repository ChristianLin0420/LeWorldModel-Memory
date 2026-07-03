# Finite Context Is Not Persistent State: A Frozen Falsification Study in a LeWorldModel-Derived JEPA

## Abstract

Pixel joint-embedding predictors can be temporally causal and still forget: evidence disappears the moment it leaves a finite context window. We test whether an explicit persistent state repairs this in a LeWorldModel (LeWM)-derived pixel JEPA. The candidate, a compact Shared-Action Shrinkage Predict--Correct (SAS-PC) module, adds two action-conditioned predict--correct states to a true sliding $H=3$ predictor; it receives no simulator state, reward, corruption label, or memory-specific objective. Before opening the cohort we froze five DeepMind Control tasks, eight separately trained designs, five seeds, 100 epochs, four held-out corruption families, and eleven conjunctive gates. The 200-cell confirmation **fails**. SAS-PC changes held-out prior-state error (positive favors SAS-PC) by **{{N_MEAN}}** versus no carrier, **{{R_MEAN}}** versus the per-cell better GRU/SSM (95% crossed-bootstrap CI **{{R_CI}}**), and **{{I_MEAN}}** versus a causally legal initial-frame/action integrator. Its action-transport intervention passes at **{{A_MEAN}}**, while the joint-read contrast is **{{J_MEAN}}**. Swimmer and Acrobot take opposite signs in all five seeds against the recurrent envelope, every integrator comparison is unfavorable, and representation-rank and convergence guards fail. The favorable no-carrier and action-transport contrasts therefore cannot be promoted to confirmed class or mechanism claims. Finite context is not persistent state, but that architectural fact is not itself a performance result.

## 1. Introduction

Joint-Embedding Predictive Architectures (JEPAs) learn dynamics by forecasting representations rather than reconstructing pixels. Recent systems add action conditioning, short video context, and latent-space planning \citep{ref1,ref2,ref3,ref4}, but their temporal states are not interchangeable, and three regimes are worth separating. A frame-local encoder carries no temporal state at all. A causal predictor over the latest $H$ latent/action tokens has *finite-context* memory: it can integrate evidence inside the window, but whatever leaves the window is unrecoverable. An episode state $m_t=F(m_{t-1},z_t,a_{t-1})$ can, in principle, *persist* evidence indefinitely. None of these properties implies causal representation learning \citep{ref5}.

Published LeWM is action-conditioned and causally masked, with configured history $H=3$ for PushT and OGBench-Cube and $H=1$ for TwoRoom \citep{ref1}; it is therefore neither memoryless nor non-causal. The narrower question this paper asks is whether an explicit state surviving beyond $H$ transports *useful* information under partial observability. Because any recurrent module beats a windowed baseline whenever old evidence has value, a candidate carrier must also survive ordinary recurrent baselines, a causally legal initial-frame/action summary, exact component interventions, and representation and convergence health checks.

We integrate compact Shared-Action Shrinkage Predict--Correct (SAS-PC) memory into a causally normalized, VICReg-style \citep{ref17} LeWM-derived pixel host. All eight designs share the encoder, the $H=3$ predictor, data, targets, optimization, and evaluation; only the persistent carrier or a named intervention changes. Because the objective replaces LeWM's SIGReg recipe \citep{ref16}, this is an architecture study, **not** evidence that SAS-PC improves original LeWM.

The result is deliberately sharper than "memory helps." Favorable finite-window and action-transport contrasts coexist with unfavorable stronger references, unsupported hierarchy contrasts, and failed validity guards; task, seed, corruption, and phase decompositions show why these statements must remain separate. Our contributions are: (i) an explicit within-grid integration separating finite context from persistent state, with the carrier's update equations and design rationale in full; (ii) a frozen $5$-task $\times$ $8$-design $\times$ $5$-seed confirmation with recurrent, integrator, intervention, health, and convergence controls; and (iii) a complete, task- and seed-resolved falsification identifying exactly which registered contrasts pass and which fail.

## 2. Related Work and Claim Scope

Finite-context JEPAs include DINO-WM, seq-JEPA, and LeWM \citep{ref1,ref2,ref3}; a Transformer over a configured block need not preserve evidence after the block is discarded. Persistent state is well established in PlaNet/Dreamer recurrent state-space models (RSSMs), recurrent and S4 world-model comparisons, and long-horizon memory studies \citep{ref6,ref7,ref8,ref9,ref10}. Multi-timescale state and predict--correct filtering predate SAS-PC in MTS3 and recurrent Kalman networks, and ELMUR and Flow Equivariant World Models provide other structured memories \citep{ref11,ref12,ref13,ref14,ref15}. We therefore claim neither the first recurrent world model nor any novelty for recurrence, filtering, action conditioning, or multiple timescales in isolation.

Our claims form a strict ladder. The information-flow graph (Figure \ref{fig:fig-v18-architecture}) establishes an **architectural** distinction between finite context and persistent state. Absence of future inputs establishes **temporal causality**, not causal representation. A registered performance contrast tests **state utility**; a separately retrained component control tests an implemented **mechanism**. None identifies environment causal structure; we evaluate the first four levels and use "causal" only for temporal information flow or an explicit model intervention. The full claim boundary is registered in Appendix G.

## 3. Architecture: A Persistent Carrier Inside a Finite-Context Host

Every design uses a per-frame Vision Transformer (ViT) encoder $E_\theta$ ($64\times64$ RGB, $8\times8$ patches, $D=128$, six layers, four heads), an action-conditioned causal Transformer predictor $P_\phi$ (four layers, eight heads), and the true sliding window of the latest $H=3$ latent/action tokens (Figure \ref{fig:fig-v18-architecture}). Two synchronized views share encoder weights: corrupted frames produce the recurrent input $z_t^c=E_\theta(o_t^c)$, while clean frames produce active, non-stop-gradient targets $z_t^\star=E_\theta(o_t)$ with dropout disabled. Per-frame feature normalization prevents future-frame or cross-example leakage; target statistics couple samples only through the loss. For valid aligned windows $\mathcal W$, the host objective is

$$
\mathcal L=\mathbb E_{(i,t)\in\mathcal W}\!\left[\tfrac1D\big\|P_\phi(\tilde z^c_{i,t-2:t},a_{i,t-2:t})-z^\star_{i,t+1}\big\|_2^2\right]+\mathcal L_{\mathrm{var}}(Z^\star)+\mathcal L_{\mathrm{cov}}(Z^\star),
\label{eq:host}
$$

with unit-weight VICReg-style variance and covariance terms applied *only to active clean targets* (exact forms in Appendix A). Regularizing targets rather than predictions stabilizes the joint-embedding fixed point without giving the memory path an auxiliary teacher: there is no memory-specific loss, no hidden-clean update, and no state, reward, or corruption-label input anywhere in the model. This VICReg-style objective differs from original SIGReg LeWM \citep{ref1,ref16,ref17}; Section 2 scopes every claim accordingly.

![LeWM-derived host and SAS-PC architecture](figures/fig_v18_architecture.png)
*SAS-PC adds an episode-persistent path while the host predictor retains a finite $H=3$ window. Corrupted and clean streams share the encoder, but only corrupted latents update memory. Green nodes are the SAS-PC predict--correct path; teal denotes clean targets and losses. Solid edges are model or training flow; the dashed strip at the bottom is the pre-correction tap that exists only for frozen post-training evaluation.*

### 3.1 The carrier: predict, correct, read

SAS-PC maintains two full-width states $m_t^k\in\mathbb R^D$, $k\in\{f,m\}$, with fixed structural timescales $\tau=(2,8)$ and rates $\beta_k=1-e^{-1/\tau_k}$. Each step factors into three stages.

**Predict: shared action transport.** A single bias-free map $W_a$ splits the last action into a multiplicative channel gate and an additive drive, $(d_{t-1},v_{t-1})=W_a\,a_{t-1}$, and extrapolates both states:

$$
p_t^k=m_{t-1}^k+\beta_k\tanh\!\big(v_{t-1}+d_{t-1}\odot\mathrm{LN}(m_{t-1}^k)\big).
\label{eq:predict}
$$

Equation \ref{eq:predict} advances each state as a bounded-drive integrator: $\beta_k$ is the discrete-time update rate of a process with time constant $\tau_k$, so the fast state ($\tau_f=2$) tracks within-window transients while the medium state ($\tau_m=8$) spans the 6--12-step training gaps and reaches toward the held-out 16--24-step freezes. The rates are fixed scalars by design: with learned rates, the rate, gate, and shrinkage parameters become mutually non-identifiable, and learned per-channel rate spectra underperformed two fixed scalars during development on disjoint tasks. The $\tanh$ bounds the per-step action displacement so blind rollouts cannot diverge, and $d_{t-1}\odot\mathrm{LN}(m_{t-1}^k)$ makes transport state-dependent rather than a constant drift; both levels share one physical $W_a$ because level-specific heads lost to a shared-action control in the same disjoint-task development. During a corruption gap this term supplies the only forward dynamics acting on the belief (correction still fires, but on a corrupted latent); it is exactly the component the no-action arm deletes, elevating action transport from a design choice to a registered, testable mechanism.

**Correct: shrinkage-gated innovation.** The current corrupted latent is projected once, $x_t=W_xz_t^c$, and each prior is corrected by its innovation $x_t-p_t^k$:

$$
\begin{aligned}
q_t^k&=\sigma\!\big(b_k+[\,w_z^\top\mathrm{LN}(z_t^c)+w_e^\top\mathrm{LN}(x_t-p_t^k)\,]/\sqrt D\big),\\
g_t^k&=(1-\rho_k)\,\sigma(b_k)+\rho_k\,q_t^k,\qquad
m_t^k=p_t^k+\beta_k\,g_t^k\,(x_t-p_t^k),\qquad \rho_k=\sigma(c_k).
\end{aligned}
\label{eq:correct}
$$

The gate $g_t^k$ plays the role of a Kalman gain \citep{ref12}, but with two deliberate restrictions. First, the correction is *rate-scaled*: multiplying by $\beta_k$ caps each level's observation intake at its structural timescale, a shrinkage that prevents a slow state from being overwritten by one noisy frame. Second, $g_t^k$ is a convex mixture of a constant "static expert" $\sigma(b_k)$, which trusts observations by a fixed amount, and an input-conditioned "dynamic expert" $q_t^k$, which reads the latent and the innovation. The learned scalar $\rho_k$ selects a point on this segment; $\rho_k\to0$ recovers purely static correction and $\rho_k\to1$ purely dynamic correction, which are exactly the frozen static and dynamic endpoint arms. Development on disjoint tasks showed static winning on some tasks and dynamic on others, so SAS-PC learns the interpolation instead of committing globally; whether the learned point beats the per-cell better endpoint is the shrinkage gate of Table \ref{tbl:gate-receipts}. The mixture also encodes a falsifiable prediction about corruption type: when observations are merely *absent* (freezes), correction is cheap and transport dominates, but when the incoming latent is itself *corrupted* (noise, checkerboard), an input-conditioned gate must learn to close; Section 5's corruption decomposition tests this.

**Read: routed residual with a null initialization.** A single time-independent softmax combines the corrected states, and a residual returns the result to the predictor stream:

$$
\tilde z_t^c=z_t^c+W_o\,\mathrm{RMSNorm}\big(\pi_fm_t^f+\pi_mm_t^m\big),\qquad \pi=\mathrm{softmax}(\ell).
\label{eq:read}
$$

The route is deliberately not a per-token router: any dynamic-memory claim must then live in action evolution or correction timing, where the interventions can test it, rather than in a confounded learned horizon selector. RMSNorm anchors the read scale so state magnitude cannot act as a shortcut, and the residual form leaves the predictor interface unchanged, making the no-carrier host an exact ablation. Initialization is identity/zero ($W_x=I$, $W_o=W_a=0$, uniform route, $\rho_k=\tfrac12$): at step zero the fused model *is* the no-carrier host, so every memory effect is learned and all paired arms start from the same prediction path; both states warm-start from $W_xz_0^c$. The module carries $2D$ persistent floats and $2D^2+2AD+2D+6$ trainable parameters --- 33,286--36,614 across task action dimensions, versus 35,048 for the width-matched GRU; recurrent carriers differ by at most 5.29% and total models by at most 0.09%.

### 3.2 Matched carriers and exact interventions

Because every stage of Equations \ref{eq:predict}--\ref{eq:read} is a separable design decision, each frozen arm deletes or pins exactly one of them (Table \ref{tbl:arms}). The GRU (gated recurrent unit) and diagonal state-space model (SSM) carriers read visual latents only, while the shared predictor remains action-conditioned for all arms; the no-action arm therefore isolates *internal* action transport within the candidate family rather than action information in general. All arms are separately trained from scratch; none is a post hoc modification of a shared checkpoint.

\begin{table}[!b]
\centering
\caption{The eight frozen arms as equation-level interventions. Every arm retrains the full model; only the named component changes. Parameter receipts are in Appendix A.}
\label{tbl:arms}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lll@{}}
\toprule
\textcolor{NVIDIADark}{\textbf{Arm}} & \textcolor{NVIDIADark}{\textbf{Change relative to Eqs.~(\ref{eq:predict})--(\ref{eq:read})}} & \textcolor{NVIDIADark}{\textbf{Question it isolates}} \\
\midrule
No carrier & carrier deleted ($\tilde z_t^c=z_t^c$) & is any episode state useful? \\
GRU & carrier $\to$ parameter-matched GRU on $z_t^c$ & does generic recurrence suffice? \\
Diag.\ SSM & carrier $\to$ learned diagonal SSM on $z_t^c$ & does linear recurrence suffice? \\
\colorbox{NVIDIAPale}{\strut SAS-PC} & none (full candidate) & the candidate \\
No action & $W_a\equiv0$ in Eq.~(\ref{eq:predict}) & internal action transport \\
Single read & $\pi=(0,1)$ pinned in Eq.~(\ref{eq:read}) & value of the two-state read \\
Static & $\rho_f=\rho_m=0$ in Eq.~(\ref{eq:correct}) & input-independent correction \\
Dynamic & $\rho_f=\rho_m=1$ in Eq.~(\ref{eq:correct}) & fully input-conditioned correction \\
\bottomrule
\end{tabular}
\end{table}

## 4. Frozen Evaluation

### 4.1 Cohort, corruptions, and grid

The unopened cohort contains Acrobot Swingup, Manipulator Bring Ball, Quadruped Run, Stacker Stack-4, and Swimmer-15 from DeepMind Control \citep{ref18} (hereafter Acrobot, Manipulator, Quadruped, Stacker, Swimmer). Each task supplies 1,200 training and 240 validation episodes of length 48 with disjoint random-action trajectories; native task state is never a training input. Training corrupts a 6--12-step interval by mean replacement or spatial cutout. Evaluation uses four *unseen* corruption families: frozen frame (freeze), Gaussian noise, checkerboard replacement, and 16--24-step long freezes. We froze $5$ tasks $\times$ $8$ designs $\times$ $5$ optimizer seeds $=200$ cells, each trained with AdamW (learning rate $3{\times}10^{-4}$, weight decay $10^{-5}$, batch 64) for exactly 100 epochs --- no early stopping, best-checkpoint selection, task or seed exclusion, rescue sweep, or architecture revision; exact optimization, parameter, and hash receipts are in Appendix A.

### 4.2 Prior-state endpoint and conservative references

After training, ridge probes ($\lambda=10^{-3}$) map each model coordinate to the native task observation; labels appear only in this frozen evaluation layer. The primary coordinate is the **prior before the current observation** --- the $H=3$ predictor prior for no carrier, the previous hidden read for GRU, the transition prior for SSM, and the action-transported routed prior for SAS-PC --- because a pre-correction prior is the only coordinate that measures information *transported through* missing observations rather than read off the current frame. Per-coordinate normalized mean-squared error (NMSE) standardizes each target dimension by its clean-training standard deviation, and the headline averages deep-gap plus first-post samples within each held-out corruption, then averages the four corruptions. For candidate error $c_{ts}$ and reference error $r_{ts}$ on task $t$ and seed $s$,

$$
\delta_{ts}=\frac{r_{ts}-c_{ts}}{\max(|r_{ts}|,10^{-12})},
\label{eq:effect}
$$

with **positive favoring SAS-PC** and tasks and seeds weighted equally. Two references are deliberately conservative. The *recurrent envelope* selects the lower of separately trained GRU/SSM per task--seed cell on the primary metric and reuses that fixed identity for deep and clean checks, so the candidate faces the stronger baseline in every cell. The *legal integrator* (Appendix E) fits the same ridge target from the candidate's own initial-frame embedding, recent and cumulative executed actions, and normalized time; it never reads a later frame, and it exists to expose how much of the endpoint a causally legal shortcut can already explain. Confidence intervals (CIs) use 100,000 crossed bootstrap draws resampling the task and seed axes independently, which preserves both generalization axes; an iid bootstrap over 25 cells would estimate a different quantity.

### 4.3 The conjunctive confirmation rule

Confirmation was registered as a *conjunction* of eleven gates spanning integrity, the recurrent/no-carrier/integrator comparisons, deep-gap persistence, the component interventions of Table \ref{tbl:arms}, a clean-prior guard, representation health (per-cell channel variance and effective rank), and late-training convergence. All thresholds, the metric, the aggregation, and the analysis code were frozen before any cohort result existed; any single miss fixes the immutable label `CONFIRMATION_FAILED`. The gates are enumerated with their frozen requirements and observed receipts in Table \ref{tbl:gate-receipts}.

## 5. Results: The Frozen Conjunction Fails

The write-once analysis validates **{{VALID}}/200** cells, all 100-epoch histories, rollouts, remote receipts, and source/cache/command hashes. Four gates pass and seven fail; the official label is **`CONFIRMATION_FAILED`**. Table \ref{tbl:gate-receipts} gives every gate receipt, and Figure \ref{fig:fig-v18-evidence} places every registered effect on one sign convention.

{{GATE_TABLE}}

![Registered V18 effect estimates](figures/fig_v18_evidence.png)
*Registered effects on one sign convention. Green points are SAS-PC estimates, gray bars are crossed task-by-seed 95% intervals, and the right columns report estimates, cell/task wins, and gate verdicts. Positive favors SAS-PC; zero is an effect reference, not every row's decision boundary. Table \ref{tbl:gate-receipts} gives the distinct frozen criteria; decisions use unrounded values.*

### 5.1 The no-carrier contrast is favorable; stronger references are not

The no-carrier effect is broad (**{{N_MEAN}}**, {{N_WINS}}/25 cells, {{N_TASKS}}/5 tasks): within this grid, episode state descriptively beats the finite window. It is not a validated class or method-ranking result, because the stronger references reverse it. The per-cell recurrent envelope yields **{{R_MEAN}}** (GRU selected {{GRU_COUNT}} times, SSM {{SSM_COUNT}}), and the legal integrator is better in **25/25 cells and 5/5 tasks** --- pooled prior-state NMSE {{SAS_POOLED_PRIMARY}} for SAS-PC against {{INTEGRATOR_POOLED_PRIMARY}} for the integrator. Even the integrator's smallest task-level advantage corresponds to a SAS-PC effect of **{{INTEGRATOR_CLOSEST_TASK}}**, the largest **{{INTEGRATOR_WORST_TASK}}**. The integrator is a diagnostic rather than a deployable world model, but the miss is neither marginal nor task-specific.

The recurrent comparison is structured rather than uniformly noisy (Figure \ref{fig:fig-v18-secondary}a). Acrobot is **{{R_ACROBOT}}** with {{R_ACROBOT_WINS}}/5 seed wins while Swimmer is **{{R_SWIMMER}}** with {{R_SWIMMER_WINS}}/5; Quadruped's task effect is only **{{R_QUADRUPED}}**, yet its cells range from **{{R_QUADRUPED_MIN}}** to **{{R_QUADRUPED_MAX}}**. The near-zero aggregate therefore combines stable task reversals with one genuinely seed-sensitive task; it is not evidence of task-invariant equivalence. Table \ref{tbl:task-nmse} reports the underlying per-task errors.

{{MAIN_TASK_TABLE}}

Clean-state quality also fails to transfer to the corrupted endpoint. Against the same fixed recurrent identities, SAS-PC gains **{{C_MEAN}}** (95% CI {{C_CI}}; {{C_WINS}}/25 cells) on the clean prior but **{{R_MEAN}}** on the registered corrupted prior. Uniform clean decodability does not establish robust missing-observation transport.

The aggregate further hides a corruption reversal (Figure \ref{fig:fig-v18-secondary}b). Freeze is **{{FREEZE_R}}** ({{FREEZE_WINS}}/25 cells, {{FREEZE_TASKS}}/5 tasks) and long freeze **{{LONG_FREEZE_R}}** ({{LONG_FREEZE_WINS}}/25, {{LONG_FREEZE_TASKS}}/5), whereas Gaussian noise is **{{GAUSSIAN_R}}** ({{GAUSSIAN_WINS}}/25, {{GAUSSIAN_TASKS}}/5) and checkerboard **{{CHECKER_R}}** ({{CHECKER_WINS}}/25, {{CHECKER_TASKS}}/5). The complete carrier's advantage is descriptively confined to replacement-style temporal freezes rather than generic corruption robustness --- the pattern Equation \ref{eq:correct} makes testable.

![Task, corruption, and phase heterogeneity](figures/fig_v18_secondary.png)
*Where the aggregate changes sign. (a) Seed cells and task-mean effects against the primary-selected recurrent identity. (b) Corruption-specific and (c) equal-condition phase effects with descriptive crossed 95% intervals; the timeline glyph defines the phases. Positive favors SAS-PC. The post phase excludes the first reappearing observation; deep gap is nested within whole gap. Panels (b, c) are unadjusted, define no new gate, and cannot change `CONFIRMATION_FAILED`.*

The phase profile sharpens the interpretation (Figure \ref{fig:fig-v18-secondary}c). Deep-gap (**{{PHASE_DEEP_MEAN}}**, {{PHASE_DEEP_WINS}}/25) and first-post (**{{FIRST_POST_MEAN}}**, {{FIRST_POST_WINS}}/25) effects are unfavorable, but the later post-gap effect is **{{POST_MEAN}}** (95% CI {{POST_CI}}; {{POST_WINS}}/25 cells, {{POST_TASKS}}/5 tasks). Descriptively, the candidate looks most useful during recovery after observations resume --- not at the deepest missing-observation or first-reappearance points where a persistent-transport story would predict an advantage.

### 5.2 Action transport passes its contrast; hierarchy contrasts do not

Removing recurrent action features (the no-action arm) worsens all five task means and 23/25 cell effects, giving **{{A_MEAN}}** (95% CI {{A_CI}}) and passing its registered contrast; its descriptive effect stays positive under every corruption, from **{{ACTION_CONDITION_MIN}}** to **{{ACTION_CONDITION_MAX}}**. The complete carrier's recurrent-baseline reversal is therefore narrower than the contribution of action transport itself: the mechanism of Equation \ref{eq:predict} meets its registered bar within this grid even where the full design does not.

The hierarchy evidence points the other way. Joint access to both states yields **{{J_MEAN}}** (95% CI {{J_CI}}) with only 6/25 wins; it is near neutral on freeze (**{{JOINT_FREEZE}}**) and long freeze (**{{JOINT_LONG_FREEZE}}**) but unfavorable on Gaussian (**{{JOINT_GAUSSIAN}}**) and checkerboard (**{{JOINT_CHECKER}}**). Learned shrinkage loses every task mean to the per-cell better fixed-$\rho$ endpoint (**{{E_MEAN}}**; dynamic selected {{DYNAMIC_COUNT}} of 25 cells, static {{STATIC_COUNT}}), so the interior point of Equation \ref{eq:correct} was not vindicated either. Consistently, the simpler single-read control has the best within-block rank profile of all eight designs --- mean rank **{{SINGLE_RANK}}** (first in {{SINGLE_FIRST}}/25, top-three in {{SINGLE_TOP3}}/25) versus **{{SAS_RANK}}** for full SAS-PC ({{SAS_FIRST}}/25; {{SAS_TOP3}}/25; Figure \ref{fig:fig-v18-task-design}, Appendix C). These ranks are descriptive, but they reinforce rather than rescue the failed component contrasts.

The optimizer-seed axis makes the same separation. Equal-task effects are positive at all five seeds for no carrier and action transport, but at only {{R_SEED_POS}}/5 seeds for the recurrent envelope and {{J_SEED_POS}}/5 for joint read. This consistency check is not a gate; it shows the favorable class and action patterns are less seed-dependent than the SAS-PC-specific and hierarchy comparisons.

### 5.3 Validity failures concentrate by task and span arms

Variance passes 200/200 cells, but effective rank passes only **{{RANK_PASS}}/200** and convergence **{{CONV_PASS}}/200**. The strongest favorable recurrent task, Swimmer (**{{R_SWIMMER}}**, 5/5 wins), has 0/40 joint guard passes; the strongest unfavorable task, Acrobot (**{{R_ACROBOT}}**, 0/5), has 40/40 (Table \ref{tbl:validity-by-task}). Yet joint-pass counts are nearly identical across designs --- {{DESIGN_BOTH_MIN}}/25 to {{DESIGN_BOTH_MAX}}/25, with SAS-PC at {{SAS_BOTH}}/25 --- so the health problem is task-structured rather than specific to one arm (Figure \ref{fig:fig-v18-task-design}). We do not filter to the 118 jointly passing cells: that would redefine the frozen estimand after seeing outcomes.

{{VALIDITY_TASK_TABLE}}

These diagnostics leave an artifact-complete falsification but block promotion of the within-grid rankings to claims about a generally superior healthy, converged system. Appendix B reports all registered contrasts in full; Appendix C gives all eight raw designs, exact ranks, and condition/phase receipts; Appendix D gives arm-level health and convergence counts.

## 6. Discussion and Limitations

**A class benefit is not a carrier benefit.** A causal $H=3$ predictor has no episode state, so the favorable no-carrier contrast is compatible with older evidence being useful. The stronger recurrent envelope asks a different question: does *this* shared-action predict--correct carrier use that opportunity better than ordinary learned recurrence? Its negative estimate prevents that promotion. The distinction matters because a finite-window baseline can make any recurrent module look like evidence for its own internal design; only the registered comparators can establish that SAS-PC is the right implementation.

**The integrator changes how the endpoint should be read.** Under random-action control trajectories, an initial visual embedding plus executed actions and time retains substantial information about native state. Losing all 25 integrator comparisons does not mean later images are useless; it means the chosen prior-state endpoint admits a strong causally legal shortcut, and SAS-PC does not extract more decodable state than that summary. Future memory benchmarks should report both a recurrent visual baseline and an initial-frame/action integrator; beating only the finite-window host leaves the central ambiguity unresolved.

**Predict--correct behavior is corruption-dependent.** Freezes preserve the last visual value while removing new evidence; Gaussian and checkerboard corrupt the current input itself. The sign reversal is consistent with the hypothesis embedded in Equation \ref{eq:correct}: action prediction helps when observations are absent, while correction can hurt when the incoming latent is unreliable and the learned gate fails to close. The phase profile sharpens this --- the favorable descriptive effect appears after observations resume, resembling improved recovery more than superior blind propagation. It remains a hypothesis because condition and phase intervals are unadjusted and the exporter did not retain per-step gate trajectories.

**Clean decodability is not persistent transport.** SAS-PC improves every clean-prior cell against the fixed recurrent identities while losing the corresponding corrupted-prior contrast. A clean objective can strengthen latent statistics or ordinary one-step prediction without preserving the right information through missing observations. Memory studies should therefore report clean prior, blind-gap prior, first-reappearance correction, and later recovery separately; a single clean linear-probe gain cannot stand in for memory robustness.

**The two-state hierarchy is not supported.** Action transport is the only component intervention meeting its registered bar, and its effect spans all four corruptions. In contrast, the single-read control has the best rank profile, the joint-read contrast is unfavorable, and learned interior shrinkage loses to the endpoint envelope. The data support neither a necessary fast/medium decomposition nor learned horizon discovery: here the extra routed read behaves as complexity without demonstrated benefit, though multiple state variables may still help in other regimes.

**Validity is part of the estimand, not a cleanup step.** Swimmer supplies the strongest favorable recurrent effect while failing the rank guard in every arm; Acrobot supplies the strongest unfavorable effect while passing everywhere. Because joint-pass counts are nearly constant across designs, the pathology is not a convenient excuse for one candidate's loss --- it is a task-level warning about what the whole grid measures. Filtering to healthy cells would answer a different question, so the negative label retains all 200 cells.

**Design implication for a new cohort.** The next study should combine an exact-SIGReg LeWM host with representation stabilization, an action-conditioned *single-state* predict--correct baseline, internally action-conditioned GRU/SSM comparators, and the same legal integrator; it should retain per-step correction gates and route weights, preregister deep-gap versus recovery-phase predictions, and evaluate executed control or planning in addition to state probes. Those changes test the narrow hypotheses exposed here --- action transport and recovery --- instead of rerunning the rejected broad claim.

Limitations remain substantial. The host uses VICReg-style clean-target regularization rather than original SIGReg LeWM; corruptions and random actions emphasize state estimation, not return. GRU matching is approximate, and the comparison does not exhaust RSSM, S4/Mamba, retrieval, or long-context baselines. SAS-PC was selected adaptively on other tasks. The exporter retained only final shrinkage coefficients and action-feature norms; **per-step gate vectors and route weights were not retained**, so no trajectory-level claim is possible. Any replay is post hoc and cannot alter the frozen result; all proposed extensions require a genuinely new cohort.

## 7. Conclusion

Finite context and persistent state are different architectural properties, but a favorable no-carrier contrast is insufficient to validate a recurrent design. Across 200 frozen cells, the no-carrier and action-transport contrasts favor SAS-PC; the stronger recurrent and integrator references do not; the hierarchy contrasts are unsupported; and the validity guards fail. The complete negative result is more informative than a baseline win: it falsifies the broad method claim on this cohort while isolating action transport --- the one mechanism that survived its own intervention --- as the hypothesis a new, valid cohort should test.

## Reproducibility Statement

\begingroup\small
The anonymous supplement contains the frozen protocol and source, all 200 cell rows, 33 registered contrasts, task/seed effect matrices, figures, and deterministic verification tools. Hash-bound private artifacts additionally retain checkpoints, histories, held-out rollouts, and remote receipts. {{RESTART_REPRO_TEXT}} Appendix F itemizes the audited interruptions. Full identities are redacted for review; the curated archive can be verified without the private repository.
\endgroup

## References

Maes, Le Lidec, Scieur, LeCun, Balestriero. *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.* 2026 (arXiv:2603.19312). · Ghaemi, Muller, Bakhtiari. *seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models.* NeurIPS 2025. · Zhou, Pan, LeCun, Pinto. *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning.* ICML 2025. · Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* 2025 (arXiv:2506.09985). · Schölkopf et al. *Toward Causal Representation Learning.* Proceedings of the IEEE, 2021. · Hafner et al. *Learning Latent Dynamics for Planning from Pixels.* ICML 2019. · Hafner et al. *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR 2020. · Hafner et al. *Mastering Diverse Control Tasks through World Models.* Nature, 2025. · Deng, Park, Ahn. *Facing Off World Model Backbones: RNNs, Transformers, and S4.* NeurIPS 2023. · Samsami et al. *Mastering Memory Tasks with World Models.* ICLR 2024. · Shaj et al. *Multi Time Scale World Models.* NeurIPS 2023. · Becker et al. *Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces.* ICML 2019. · Shaj et al. *Action-Conditional Recurrent Kalman Networks for Forward and Inverse Dynamics Learning.* CoRL 2020, PMLR 155, 2021. · Cherepanov, Kovalev, Panov. *ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL Problems.* ICLR 2026. · Lillemark et al. *Flow Equivariant World Models: Structured Memory for Dynamic Environments.* ICML 2026 (arXiv:2601.01075). · Balestriero, LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* 2025 (arXiv:2511.08544). · Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022. · Tassa et al. *DeepMind Control Suite.* 2018 (arXiv:1801.00690).

## Appendix A. Exact protocol and analysis

### A.1 Objective terms, initialization, and parameter matching

The variance and covariance terms of Equation \ref{eq:host} act on the matrix $\bar Z^\star$ of batch-and-time-centered active clean targets with $N$ rows:

$$
\mathcal L_{\mathrm{var}}(Z^\star)=\frac1D\sum_{d=1}^{D}\max\!\big(0,\,1-\sqrt{\mathrm{Var}(z^\star_{\cdot d})+\varepsilon}\big),\qquad
\mathcal L_{\mathrm{cov}}(Z^\star)=\frac1D\sum_{d\ne d'}C_{dd'}^2,\quad C=\frac{\bar Z^{\star\top}\bar Z^\star}{N-1}.
\label{eq:vicreg}
$$

All three loss terms have unit weight; there is no selectable regularizer coefficient. Clean-frame activations, simulator state, rewards, and corruption labels or masks never enter the recurrent update; there is no memory teacher and no memory-specific loss. SAS-PC initialization is $W_x=I$, $W_o=W_a=0$, $b_f=b_m=2$, $c_f=c_m=0$, uniform route. The GRU width is fixed once by minimizing $|4Dh+3h^2+6h-(2D^2+16D)|$ over hidden sizes $h\le D$, never re-selected per task.

### A.2 Immutable identities

The frozen execution identities are:

{{ARTIFACT_TABLE}}

The four-GPU queue was fixed as GPU 0: Acrobot then Stacker; GPU 1: Manipulator; GPU 2: Quadruped; GPU 3: Swimmer. Every task used optimizer seeds 18001--18005. The protocol stores all 200 expanded commands and per-source SHA-256 values; the final analyzer revalidates them before writing results.

### A.3 Metric phases

For target time $t$ and corruption interval $[b,e)$, `gap` is $b\le t<e$, `deep` is $b+H\le t<e$, `first_post` is $t=e$, and `post` is $e<t\le e+H$. The primary held-out metric selects deep and first-post samples. This selection, the four-corruption average, and all probe splits were frozen before launch.

### A.4 Crossed bootstrap

Let $\Delta\in\mathbb R^{5\times5}$ contain task-by-seed paired effects. Each draw samples five task indices and five seed indices independently with replacement, takes their Cartesian $5\times5$ submatrix, and averages all entries. We use 100,000 PCG64 draws with seed 18018 and linear 2.5/97.5 percentiles, preserving the two crossed generalization axes.

### A.5 Comparator identity

For each task--seed cell, the recurrent identity is the lower primary held-out prior NMSE of GRU and SSM; exact ties select GRU. The same identity supplies deep and clean references; selecting a new best model separately for those metrics is prohibited. Static/dynamic identity is selected per cell only for the endpoint noninferiority contrast.

### A.6 Frozen arms and parameter receipts

The eight arms of Table \ref{tbl:arms} are separately trained; GRU/SSM update from visual latents only, although the shared predictor remains action-conditioned. SAS-PC has 33,286--36,614 carrier parameters across action dimensions; the fixed GRU width gives 35,048. Recurrent carrier sizes differ by at most 5.29% and total models by at most 0.09%.

## Appendix B. Registered contrasts in full

All eight registered primary contrasts with crossed 95% and 90% intervals and pooled absolute errors:

{{CONTRAST_FULL_TABLE}}

The table is generated directly from `confirmation_analysis.json`; decisions use full precision, so rounding here cannot change a gate. All 200 auditable cell rows, the 33 registered contrast rows, the $5\times5$ cell-effect matrices, and selected identities are supplied in `confirmation_cells.csv`, `confirmation_contrasts.csv`, and `confirmation_analysis.json`.

## Appendix C. Full results

Table \ref{tbl:task-nmse} and Table \ref{tbl:component-controls} jointly report all eight trained designs. Values are held-out prior-state NMSE as mean $\pm$ standard deviation over five optimizer seeds; lower is better.

{{TASK_DESIGN_TABLE}}

{{CLEAN_TASK_TABLE}}

![Within-block design-rank distributions and validity structure](figures/fig_v18_task_design.png)
*(a) Within-task, within-seed rank distributions for all eight designs; light points are the 25 task--seed ranks, bars span the interquartile range, ticks mark medians, diamonds mark mean rank, and rank 1 is the lowest NMSE within a block. SAS-PC is highlighted only to locate the candidate; ranks are descriptive, pool no raw NMSE across tasks, and define no superiority test. (b) Joint validity-guard passes per task and design (of five seeds), showing that guard failures are task-structured rather than design-specific.*

Exact rank summaries are:

{{DESIGN_RANK_TABLE}}

The frozen recurrent-envelope identity counts by task are:

{{TASK_RECURRENT_TABLE}}

Condition-specific component effects and equal-condition phase slices provide secondary context for Figure \ref{fig:fig-v18-secondary}:

{{SECONDARY_COMPONENT_TABLE}}

{{SECONDARY_PHASE_TABLE}}

Per-corruption and per-phase values remain in each cell's `metrics.json` and held-out rollout arrays. The paper does not pool raw task-state MSE across heterogeneous tasks.

## Appendix D. Representation and convergence

{{REPRESENTATION_TABLE}}

Representation health requires both thresholds to be met in every cell. Convergence is the absolute relative change between mean validation predictive loss over epochs 81--90 and 91--100. The gate requires every cell's late change to be at most 5%; no failing arm is removed.

## Appendix E. Integrator guard

For target $t$, the legal feature vector is $[E_\theta(o_0),a_{t-3:t-1},\sum_{j=0}^{t-1}a_j,t/(L-1)]$, with zero-padding before three actions exist. The candidate checkpoint supplies $E_\theta$; a ridge map is fit on clean training targets and evaluated only on deep and first-post held-out samples. The integrator receives no observation after the visible $o_0$.

{{INTEGRATOR_TABLE}}

## Appendix F. Adaptive provenance and restart

SAS-PC was selected after V1--V7 development on different tasks. Its earlier adaptive V8 study completed 325 cells but had immutable label `PILOT_NO_GO_FINAL_DESCRIPTIVE`; it cannot rescue the present confirmation (V18). V9--V17 are likewise development/host audits, not confirmation. {{RESTART_APPENDIX_TEXT}} The registered commands, seeds, source hashes, and stopping rule were unchanged.

## Appendix G. Claim boundary

The study can establish only a persistent-state or component-intervention result for a VICReg-trained LeWM-derived finite-context host on the frozen corruption cohort. It cannot establish improvement to original SIGReg LeWM, executed-return or planning gains, causal discovery, learned semantic hierarchy, calibrated uncertainty, learned horizon discovery, or robustness beyond these tasks and corruptions.

## Appendix H. LLM Usage Statement

OpenAI Codex assisted with code review, experiment monitoring, artifact auditing, deterministic result-to-manuscript tooling, and manuscript drafting/editing. The authors verified the executed code, artifacts, statistics, citations, and final claims and retain responsibility for the work.
