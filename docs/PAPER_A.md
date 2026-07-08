# Beyond the Context Window: A Frozen-Host Audit of Persistent Memory in SIGReg LeWM

## Abstract

Latent world models can predict local dynamics while forgetting evidence after it leaves their observation window. Using the released SIGReg LeWorldModel (LeWM) Reacher checkpoint, we audit representation availability, causal retention, finite-context access, and rollout competence. All released components remain frozen while parameter-matched GRU, LSTM, diagonal state-space, and fixed-trust predict--correct carriers are optimized through the next-latent loss. Transient-marker and drifting-color recall pass the representation criterion; occluded-target prediction reaches only $R^2=0.010$ and is excluded at the pre-specified $.300$ threshold. Across the admitted tasks, fixed-trust trails the GRU by -0.011 accuracy (95\% CI $[-0.021,-0.001]$). At $H=56$, raw context clears the availability criterion on 2/2 tasks, whereas predictor output is interval-resolved above chance on 0/2; rollout competence passes in 4/4 task--objective cells. Overshooting does not yield an interval-resolved MSE improvement on Transient-marker recall and improves normalized MSE on Drifting-color recall. These results isolate memory effects under a common official host; memory-conditioned control is outside their scope.

## 1. Introduction

Joint-embedding world models learn dynamics without reconstructing every pixel. LeWM trains a visual encoder and an action-conditioned latent predictor end to end with SIGReg, while DINO-WM predicts over frozen DINOv2 features \citep{r_lewm,r_dinowm,r_dinov2}. V-JEPA 2-AC and recent controlled JEPA studies likewise show that compact latent prediction can support physical planning \citep{r_vjepa2,r_jepawm}. These systems have an attractive interface: encode a short history, predict future representations under candidate actions, and plan with a task-specific objective. That interface is not, by itself, persistent memory.

The distinction matters whenever a decisive observation occurs before the predictor's left context boundary. A model may attain low one-step latent error because the visible scene is locally smooth, even though a cue that no longer affects pixels has become unrecoverable. Increasing the rollout horizon does not resolve this issue: a rollout trace extrapolates from the model's current input, whereas a persistent state must first preserve evidence across real observations. Conversely, adding recurrence does not prove that the recurrence stores the variable of interest or that the predictive objective has any incentive to store it.

We isolate these questions in the released SIGReg LeWM Reacher model. Its 192-dimensional latent predictor consumes three observations and 10-dimensional blocks of five Reacher actions. During the carrier comparison, the encoder--projector, action encoder, predictor, and prediction projection remain byte-identical. Only a causal carrier is trained: it reads each real latent, exposes a pre-observation state read, and fuses its posterior read into the unchanged predictor. The intervention asks what each carrier makes linearly usable under one common host.

Before memory training, a fixed readout tests whether the encoder--projector makes each target available. The admitted categorical tasks then enter the carrier swap and a separate context-length control; a continuous future-location task that fails this criterion remains diagnostic only. Learned rollouts are evaluated independently against copy-last and shuffled-action references.

Our contributions are an official-checkpoint frozen memory intervention with matched recurrent baselines, a context sweep that separates raw access from predictor exposure, and an action-sensitive rollout test. Together they locate failures under a specified host and legal readout rather than collapsing them into one score.

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{figures/fig_a_arch.pdf}
\caption{Frozen-host persistent-memory intervention. (a) Frames pass through the released SIGReg LeWM encoder--projector. Standardized raw 10-D action blocks branch directly to the carrier, while the frozen action encoder supplies action tokens to the frozen predictor and output projection (blue dashed blocks). Only the carrier is trainable (green); the cached raw next latent supplies the target, and gradients traverse the frozen prediction path to the carrier. (b) The carrier exposes a prior read before seeing the current latent, then corrects its episode state and supplies a residual posterior read to the frozen predictor. The episode state continues after evidence has left the three-token context.}
\label{fig:architecture}
\end{figure}

## 2. Memory Claims in Finite-Context World Models

### 2.1 Context, rollout, and persistent state

A finite-context predictor reconstructs its state from the latest $H$ observations. A rollout instead extrapolates one candidate action sequence. Persistent state is updated across real observations and remains available after an event is absent from every other legal input.

LeWM and DINO-WM predict over bounded latent histories \citep{r_lewm,r_dinowm}; Fast-LeWM predicts action-prefix horizons in parallel without an episode state updated by later observations \citep{r_fastlewm}. DreamerV3, Recall to Imagine, and S4WM study recurrent or state-space dynamics, while PERSIST maintains an evolving geometric world state \citep{r_dreamerv3,r_r2i,r_s4wm,r_persist}. These interfaces are complementary, but feature prediction, state decodability, and causal use are distinct endpoints.

### 2.2 Four clocks

We distinguish four temporal scales:

1. The **observation clock** is one encoded frame; five simulator actions form each 10-D action block.
2. The **context clock** is the number $H$ of real latent tokens; the released host uses $H=3$.
3. The **evidence-age clock** counts observations since the informative event.
4. The **imagination clock** is rollout horizon $K$ after prediction begins.

Increasing $H$ postpones context expiry; increasing $K$ increases extrapolation. A carrier changes the evidence-age clock only when it is updated through real observations and read after the cue expires from the raw window.

\begin{table}[!t]
\centering
\fontsize{7.2}{8.0}\selectfont
\setlength{\tabcolsep}{2.0pt}
\renewcommand{\arraystretch}{1.08}
\caption{Temporal-interface checklist. A check denotes an explicit interface or experiment, not a cross-paper performance ranking.}
\label{tab:interfaces}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}*{5}{>{\centering\arraybackslash}p{0.105\linewidth}}Y@{}}
\toprule
System & \shortstack{Bounded\\obs. context} & \shortstack{Episode\\state} & \shortstack{State carried\\beyond $H$} & \shortstack{Pre-frame\\read} & \shortstack{Frozen-host\\swap} & \shortstack{Multi-step\\imagination} \\
\midrule
LeWM \citep{r_lewm} & \TblPass & \TblNA & \TblNA & \TblNA & \TblNA & \TblPass \\
\rowcolor{TableGray}
Fast-LeWM \citep{r_fastlewm} & \TblPass & \TblNA & \TblNA & \TblNA & \TblNA & \TblPass \\
DINO-WM \citep{r_dinowm} & \TblPass & \TblNA & \TblNA & \TblNA & \TblNA & \TblPass \\
\rowcolor{TableGray}
Recall to Imagine \citep{r_r2i} & \TblNA & \TblPass & \TblPass & \TblNA & \TblNA & \TblPass \\
PERSIST \citep{r_persist} & \TblPart & \TblPass & \TblPass & \TblNA & \TblNA & \TblPart \\
\midrule
LeWM + persistent carrier (ours) & \TblPass & \TblPass & \TblPass & \TblPass & \TblPass & \TblNA \\
\rowcolor{TableGray}
LeWM predictor fine-tuning (ours) & \TblPass & \TblNA & \TblNA & \TblNA & \TblNA & \TblPass \\
\bottomrule
\end{tabularx}
\vspace{1pt}
\parbox{\linewidth}{\fontsize{6.8}{7.5}\selectfont Check marks denote explicit support; triangles denote a qualified, different state contract; dashes mean not reported. ``Carried'' denotes the interface, not verified target retention. Our predictor control freezes the encoder--projector, fine-tunes dynamics, and has no carrier.}
\end{table}

### 2.3 Position within LeWM-style world models

The intervention complements, rather than replaces, existing world-model designs. Original LeWM jointly learns representations and action-conditioned dynamics with SIGReg; DINO-WM predicts over externally frozen visual features \citep{r_lewm,r_dinowm}. Our frozen host serves a different purpose: it fixes both the representation and consumer so that carrier variants share one causal environment. Fast-LeWM improves how future action prefixes are queried, but parallel prediction after a planning anchor cannot recover an event already absent from that anchor's context \citep{r_fastlewm}. Recall to Imagine and S4WM instead ask how recurrent or state-space dynamics affect end-to-end control \citep{r_r2i,r_s4wm}. PERSIST uses an explicit geometric state with a different observation and update contract \citep{r_persist}.

These distinctions determine the claim. A high downstream return can combine perception, memory, dynamics, and policy improvements; a decodable recurrent state can remain unused; and a long context can expose a cue without compressing it into the predictor output. We therefore compare interfaces in Table \ref{tab:interfaces} but compare scores only within the frozen LeWM experiment. The target is not a new benchmark-leading framework. It is an attribution test for whether an episode state adds information after LeWM's legal context has expired, and whether any observed limitation belongs upstream in representation, inside persistence, or downstream in learned dynamics.

## 3. Fail-Closed Audit and Frozen Intervention

### 3.1 Claim ladder

The audit asks four ordered questions. **Demand:** does the target require an earlier observation? **Availability:** can a fixed readout recover it from the frozen representation at evidence time? **Retention:** is it recoverable from a causal carrier read after leaving raw context? **Use:** can a matched consumer exploit that state? Learned control additionally requires competent dynamics. Failure at one stage stops downstream interpretation.

If a cue disappears without affecting later pixels, next-latent loss can be minimized without retaining its class. A negative carrier result then diagnoses objective alignment under this host, not recurrent capacity. If longer raw context restores readout while its predictor output does not, the limitation lies after raw access.

### 3.2 Common carrier contract

All carrier variants implement one interface. Let $E$ be the frozen encoder--projector and $P$ the frozen predictor plus output projection. With latent $z_t$, standardized action block $a_{t-1}$, and episode state $m_{t-1}$,

$$
\begin{aligned}
z_t &= E(o_t), &
(b_t,m_t,\widetilde z_t) &= C_\phi(m_{t-1},z_t,a_{t-1}), \\
\widehat z_{t+1} &= P(\widetilde z_{t-H+1:t},a_{t-H+1:t}), &
H &= 3.
\end{aligned}
$$

The prior $b_t$ is read before consuming $z_t$; the corrected read is fused into $\widetilde z_t$. Zero-initialized read projections make every variant start as the unchanged host. Training uses only next-latent error against the raw cached target; labels never reach the carrier or predictor.

Fixed-trust predict--correct memory maintains channelwise mean and uncertainty. Actions predict the next state; an input-independent learned trust controls correction by the current latent. A diagonal retention spectrum and residual projection return the read to LeWM space, following recurrent predict--correct estimators \citep{r_rkn}.

Baselines are an action-conditioned GRU, LSTM, and diagonal state-space carrier. Their selected widths yield 75,924, 76,372, and 76,032 parameters, versus 76,032 for fixed-trust. Optimizer, updates, data, frozen host, injection point, read convention, and initialization are shared. The no-carrier contrast asks whether a recurrent path changes final information at all; cross-carrier contrasts compare learned update rules under approximately matched capacity. Both estimands are conditional on one released checkpoint and intentionally exclude end-to-end encoder--predictor co-adaptation.

## 4. Experimental Design

### 4.1 Released SIGReg host and data contract

We use the released official LeWM Reacher checkpoint \citep{r_lewm}. A ViT-Tiny with 14-pixel patches at $224\times224$ projects its class token to 192 dimensions. Six conditional transformer blocks consume three latent/action tokens before the released prediction projection.

All tasks use DeepMind Control Suite Reacher \citep{r_dmc}: 64 rendered observations per episode, with five independently sampled 2-D actions flattened between frames and standardized on the training split. Each task has 1,200 training and 240 fixed validation episodes.

**Transient-marker recall** briefly fills one of four persistent marker outlines and adds a class-colored border. Post-cue overlays are class-independent; the target is marker identity at the final decision. **Drifting-color recall** flashes the class as the color of a stochastically moving sprite, which then continues in neutral gray. **Occluded-target prediction** repeats the last visible frame for 16--20 observations while a stochastic two-dimensional target continues moving; the target is its normalized location when observations resume.

\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{figures/fig_a_protocol.pdf}
\caption{Memory tasks and frozen-representation admission. Categorical rows show the cue, class-independent delay, and last legal pre-decision frame; their evidence age is 43--53 frames. The continuous row shows the last pre-gap observation, a 16--20-frame frozen interval, and an evaluation-only outcome after the cutoff. Four cue-window latents yield categorical accuracy above $.750$; four pre-gap latents yield only $R^2=0.010$ for the continuous target, below $.300$.}
\label{fig:protocol}
\end{figure}

### 4.2 Admission tests

For categorical tasks, cue identity changes only cue-window overlays; post-cue overlays are class-independent, so the final three frames cannot reveal the class. Their cue-to-read evidence age is 43--53 frames. For the continuous task, a 16--20-frame repeated visual interval separates the last observed target from its later position.

Availability is measured before memory training. Multinomial logistic regression reads four frozen cue-window latents for categorical tasks, with $.750$ validation accuracy required against $.250$ chance. Target-standardized ridge regression reads the last four pre-gap latents for occluded-target prediction, with $R^2\ge .300$ required. The categorical tasks proceed to carrier evaluation; the continuous task is excluded because its frozen representation does not meet this predictive-sufficiency threshold.

### 4.3 Training and statistics

Carriers train for 100 AdamW epochs over five model/optimizer seeds. At held-out decision observation $q=63$, the evaluation probe concatenates $z_{60:62}$ with causal prior $b_q$; $z_q$ is excluded and no temporal pooling is used. A deterministic standardized logistic model is fit on the training split and evaluated on validation episodes without carrier gradients.

For $H\in\{3,16,32,56\}$, the encoder--projector stays frozen while positional embeddings are interpolated and the action encoder, predictor, and output projection train for 60 epochs. At cutoff $q$, a seed-independent **raw-access readout** uses the legal window's mean and last latent; a **predictor readout** uses one contextual prediction over three seeds. These feature maps are dimension-stable across $H$ but differ from the carrier probe.

The same predictor components train for 60 epochs with one-step prediction or eight-step overshooting. From fixed anchor $t=24$, three seeds recursively predict targets at $K\in\{1,2,4,8,16\}$. We report variance-normalized latent MSE, its paired ratio to copy-last, and advantage over shuffled actions. Competence requires beating copy-last with positive action advantage at every horizon through eight.

Intervals are deterministic 95\% percentile bootstraps over seeds with 20,000 draws. Carrier contrasts pair seeds and weight tasks equally. The experiment matrix and analysis rules were fixed before formal aggregation and are released with the artifacts.

## 5. Does Persistent State Add Information Beyond Three Frames?

### 5.1 Representation availability establishes the admissible scope

Figure \ref{fig:protocol} admits only the two categorical tasks: transient-marker and drifting-color accuracy are 1.000 and 1.000. Occluded-target prediction remains an availability diagnostic; its failure does not establish absence of the current pre-gap state.

### 5.2 Frozen-host carrier swap

Table \ref{tab:frozen-results} and Figure \ref{fig:memory-results}a report the carrier comparison. On transient-marker recall, no carrier obtains 0.258, fixed-trust 0.278 [0.267, 0.287], and GRU 0.274 [0.267, 0.280]. On drifting-color recall, they obtain 0.212, 0.202 [0.198, 0.204], and 0.227 [0.216, 0.239], respectively.

The pre-specified primary contrast is fixed-trust memory minus the parameter-matched GRU, paired by seed and weighted equally across tasks. Its mean is -0.011 with 95\% interval $[-0.021,-0.001]$ and 3/10 positive seed--task pairs. The corresponding contrasts against LSTM and the diagonal state-space carrier are -0.009 [-0.017, +0.001] and -0.016 [-0.024, -0.007]. Under the equal-task paired estimand, fixed-trust trails the GRU (the paired interval is wholly negative); it trails the diagonal state-space carrier (the paired interval is wholly negative).

Figure \ref{fig:memory-results}a makes the aggregation transparent. Fixed-trust is numerically positive against all learned comparators on transient-marker recall, but every task-specific interval crosses zero. On drifting-color recall all three contrasts are negative. Equal task weighting therefore prevents the marker task from hiding the color failure. Relative to no carrier, the GRU changes final accuracy by +0.016 [+0.009, +0.022] on Transient-marker recall and +0.015 [+0.003, +0.027] on Drifting-color recall; the diagonal SSM changes it by +0.016 [+0.003, +0.026] and +0.026 [+0.018, +0.033]. Fixed-trust reverses by task: +0.020 [+0.009, +0.028] versus -0.011 [-0.014, -0.008]. The recurrent paths therefore produce small, task-dependent changes, while absolute final accuracies remain close to four-way chance.

Next-latent error is reported beside accuracy because local prediction and remote retention need not agree. Host, data, context, loss, optimizer, and readout remain fixed while only recurrent state changes.

The loss ranking sharpens this point. The diagonal state-space carrier gives the lowest pooled next-latent MSE, followed by the GRU, LSTM, fixed-trust memory, and no carrier, yet all equal-task accuracies remain close to $.250$ chance. Thus reduced local loss does not establish persistent target information. The exploratory trajectory-average readout reveals a different localization. For Transient-marker recall, fixed-trust changes from 0.278 at the final causal endpoint to 0.454 with temporal aggregation, and the SSM changes from 0.274 to 0.506. On Drifting-color recall, the corresponding trajectory values are 0.408 and 0.421. The aggregate post-cue trajectory is therefore more linearly decodable than the final feature for these carriers. Because temporal support and feature maps differ, this does not prove exposure at any individual state or decision-time memory (Appendix Figure \ref{fig:app-probe}).

\begin{table*}[!t]
\centering
\fontsize{7.2}{8.0}\selectfont
\setlength{\tabcolsep}{4.0pt}
\renewcommand{\arraystretch}{1.10}
\caption{Frozen-host carrier comparison. Entries are mean [95\% seed-bootstrap CI] over five paired seeds. The probe concatenates the final three legal raw latents with the pre-observation carrier read; no carrier is deterministic.}
\label{tab:frozen-results}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.20\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}*{3}{>{\centering\arraybackslash}p{0.145\textwidth}}Y@{}}
\toprule
Carrier & Trainable params & Transient-marker $\uparrow$ & Drifting-color $\uparrow$ & Equal-task mean $\uparrow$ & Next-latent MSE $\downarrow$ \\
\midrule
No persistent carrier & 0 & 0.258 & 0.212 & 0.235 & 0.3028 \\
\rowcolor{TableGray}
Action-conditioned GRU & 75,924 & 0.274 [0.267, 0.280] & 0.227 [0.216, 0.239] & 0.251 [0.242, 0.260] & 0.1182 \\
Action-conditioned LSTM & 76,372 & 0.272 [0.258, 0.285] & 0.226 [0.211, 0.237] & 0.249 [0.238, 0.258] & 0.1258 \\
\rowcolor{TableGray}
Diagonal state-space carrier & 76,032 & 0.274 [0.262, 0.284] & 0.238 [0.231, 0.246] & 0.256 [0.249, 0.263] & 0.1019 \\
\addlinespace[1pt]
Fixed-trust predict--correct (ours) & 76,032 & 0.278 [0.267, 0.287] & 0.202 [0.198, 0.204] & 0.240 [0.235, 0.244] & 0.1404 \\
\bottomrule
\end{tabularx}
\end{table*}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig_a_evidence.pdf}
\caption{Persistent state versus finite context. (a) Paired fixed-trust-minus-comparator accuracy: translucent points are seed differences, colored markers are task means, diamonds are equal-task means, whiskers are 95\% intervals, and zero marks parity. (b) Accuracy versus context $H$: dashed curves read raw legal context; solid curves read a trained predictor output with 95\% intervals. The context sweep is a capability control, not a frozen-carrier intervention.}
\label{fig:memory-results}
\end{figure*}

### 5.3 Longer context separates access from learned exposure

Figure \ref{fig:memory-results}b compares raw access with a predictor trained at each context. For transient-marker recall, raw accuracy changes from 0.229 at $H=3$ to 0.771 at $H=56$, while predictor accuracy changes from 0.261 [0.254, 0.267] to 0.253 [0.250, 0.258]. For drifting-color recall, raw accuracy moves from 0.171 to 0.825, versus 0.208 [0.204, 0.212] to 0.190 [0.188, 0.192] for the predictor.

The raw curve asks whether the event is readable once it re-enters legal context; the predictor curve asks whether next-latent training exposes it. At $H=56$, the raw window for Transient-marker recall clears the 0.750 availability criterion, while its predictor output is not interval-resolved above chance; for Drifting-color recall, the raw window clears the 0.750 availability criterion, while its predictor output is interval-resolved below chance. Because the predictor and readout are re-estimated at each $H$, this is a capability control rather than an isolated carrier effect. The pattern is consistent with, but does not prove, an objective or predictor bottleneck.

Coverage receipts explain the discontinuity. Among the tested context lengths, at $H\in\{3,16,32\}$ no validation window contains a cue frame; at $H=56$, every validation episode contains some cue evidence, averaging about five cue frames. The $H=32$ to $H=56$ transition makes the dissociation explicit: Transient-marker recall raw access rises from 0.263 to 0.771, while local MSE worsens by 32\%; Drifting-color recall raw access rises from 0.196 to 0.825, while local MSE worsens by 26\%. Selecting context by minimum local prediction error would therefore prefer a window in which the cue is unreachable (Appendix Figure \ref{fig:app-context}).

## 6. Learned Rollout Competence

Persistent memory and imagination solve different temporal problems. We evaluate learned dynamics separately before any memory-conditioned control claim.

Figure \ref{fig:rollouts}a reports paired latent-MSE ratios to copy-last; values below one improve on repeating anchor $t=24$. Figure \ref{fig:rollouts}b reports variance-normalized shuffled-action MSE minus true-action MSE; positive values indicate useful action conditioning.

At horizon eight, the one-step objective obtains MSE/copy-last ratios of 0.043 on Transient-marker recall and 0.329 on drifting-color recall. Eight-step overshooting obtains 0.042 and 0.241, respectively. True-action advantages at the same horizon are 1.258 and 0.766 for one-step training, versus 1.226 and 0.770 for overshooting.

The full horizon profile is more informative than a single endpoint. All error ratios stay below one through diagnostic $K=16$, and action advantage grows rather than collapsing toward zero. Overshooting sacrifices short-horizon accuracy, is tied with one-step training at $K=8$ for transient-marker recall, and helps drifting-color recall at longer horizons. At $K=16$, overshooting reduces normalized latent MSE on Transient-marker recall (paired difference -0.0267 [-0.0463, -0.0154]) but increases pose MAE (+0.0375 [+0.0191, +0.0664]). On Drifting-color recall, both paired differences indicate improvement: -0.1220 [-0.1482, -0.0979] for latent MSE and -0.0634 [-0.0720, -0.0486] for pose MAE. Thus a latent-MSE improvement is not a uniform proxy for physical-state accuracy (Appendix Figure \ref{fig:app-rollout}).

The through-eight criterion passes in 4/4 task--objective cells and 12/12 trained models. This establishes dynamics competence under the two references, not cue retention or memory-conditioned control.

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/fig_a_results.pdf}
\caption{Learned-rollout competence from fixed anchor $t=24$. (a) Paired normalized-MSE ratio to copy-last on a log scale; lower is better and one marks parity. (b) Variance-normalized true-action advantage over shuffled actions; positive is better. Colors identify tasks, line style identifies one-step versus eight-step overshooting, and ribbons are 95\% seed intervals.}
\label{fig:rollouts}
\end{figure*}

## 7. What the Audit Localizes


\begin{table}[!ht]
\centering
\fontsize{7.2}{8.0}\selectfont
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.08}
\caption{Cross-experiment localization matrix. Values are target accuracy unless marked otherwise; rollout entries show one-step / overshooting status with three seeds per status. The table aligns distinct estimands rather than treating them as directly comparable scores.}
\label{tab:localization}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.25\linewidth}*{3}{>{\centering\arraybackslash}p{0.17\linewidth}}Y@{}}
\toprule
Evidence stage & Transient-marker & Drifting-color & Occluded-target & Interpretation \\
\midrule
Cue-time frozen representation & 1.000 & 1.000 & $R^2=0.010$ & Admission \\
\rowcolor{TableGray}
Final fixed-trust read & 0.278 [0.267, 0.287] & 0.202 [0.198, 0.204] & Excluded & Fixed-trust endpoint \\
Raw context at $H=56$ & 0.771 & 0.825 & -- & Finite access \\
\rowcolor{TableGray}
Predictor output at $H=56$ & 0.253 [0.250, 0.258] & 0.190 [0.188, 0.192] & -- & Learned exposure \\
Rollout gate (one-step / overshoot) & 3/3 / 3/3 & 3/3 / 3/3 & -- & Dynamics only \\
\bottomrule
\end{tabularx}
\end{table}

The experiments give three counterexamples to common memory proxies. First, low next-latent loss does not imply final semantic retention. Once a class stops affecting future latents, the next-latent objective provides no predictive pressure to preserve it. In the equal-task aggregate, the diagonal SSM has the lowest next-latent MSE (0.1019) while its final accuracy is only 0.256 [0.249, 0.263], whose interval overlaps four-way chance. Better local prediction therefore does not establish robust delayed retention. The trajectory-average diagnostic further separates aggregate post-cue decodability from a stable final read.

Second, placing a cue inside legal context does not imply that the trained predictor exposes it. At $H=56$, raw access clears the pre-specified criterion on 2/2 tasks and predictor output is resolved above chance on 0/2; at frozen $H=3$, fixed-trust is interval-resolved above no carrier on Transient-marker recall and is interval-resolved below no carrier on Drifting-color recall. The $H=56$ raw gain and near-chance predictor output separate finite access from learned compression, while the feature-map mismatch prevents a causal bottleneck claim.

Third, rollout competence and memory retention use different evidence windows. The rollout test begins from the local anchor $t=24$ and never evaluates retention of evidence preceding that anchor. Passing its copy-last and action-shuffle references therefore establishes local dynamics competence, not episode memory, planning, or control.

The cross-task reversal is itself diagnostic. Fixed-trust is +0.020 [+0.009, +0.028] relative to no carrier on transient-marker recall but -0.011 [-0.014, -0.008] on drifting-color recall, whereas the SSM is positive on both tasks. The exploratory trajectory-average feature changes the ordering again. Carrier rankings therefore vary across tasks and endpoints, and cannot be explained by parameter count alone.

For LeWM-style systems, expanding context, adding recurrence, and improving rollout objectives repair different interfaces. A stronger system may need delayed-evidence supervision, a consumer trained to use state, or end-to-end adaptation. Each changes the estimand and should be evaluated separately against this frozen diagnosis.

## 8. Limitations and Reproducibility

**Scope.** The study uses one Reacher checkpoint and three overlays in one rendered scene family. Its four-way tasks do not cover associative retrieval, spatial mapping, credit assignment, or capacity scaling; the continuous task yields no carrier ranking. Results are a controlled test of persistent cue information, not a benchmark-wide architecture ordering.

**Task naturalism and capacity.** The categorical targets are synthetic overlays chosen to separate cue time from decision time. This gives precise control over leakage, but natural control tasks may couple memory variables to later observations in ways that supply richer predictive gradients. Each episode stores one four-way item; we do not vary the number of items, distractor density, cue order, or retrieval queries. The results therefore characterize persistence over delay, not memory capacity or interference.

**Readout and estimand.** Availability and retention are relative to fixed linear readouts; linear failure is not information absence, and success is not consumer use \citep{r_belinkov,r_ravich}. Frozen swaps isolate recurrent paths but omit end-to-end co-adaptation. Context predictors and their readouts are re-estimated, the no-carrier reference has zero memory parameters, and inference cost is not matched across recurrent and transformer-context systems.

**Optimization and endpoint.** Learned carriers share one schedule and approximately matched parameter counts, but they are not separately hyperparameter-tuned and their sequential compute differs. More importantly, training uses only the official next-latent objective. GRU and diagonal SSM yield small paired gains over no carrier on both tasks, but no tested carrier has a final-read interval clearly above chance on both; this endpoint ranks what the objective learns, not what each architecture could store under delayed supervision. Our primary probe reads $b_{63}$ before the decision observation and deliberately excludes temporal aggregation. Higher decodability from an exploratory trajectory-average feature is therefore not credited as persistent decision-time memory and does not alter the headline endpoint.

**Replication.** Seeds reuse fixed splits, so intervals measure optimization and initialization variation rather than new environments. Five seeds evaluate carriers and three evaluate context and rollout models; only one rollout anchor is used. Future work should regenerate episodes, vary scenes and physics, test more checkpoints, and sample anchors. We verify exact checkpoint loading and frozen-host identity by hashes; figures and tables come from one validated seed-level aggregate.

The study also does not equalize wall-clock cost. Recurrent carriers process all 64 observations sequentially, whereas context predictors attend over exact-$H$ windows whose count changes with $H$. The continuous task's failed admission can reflect stochastic future uncertainty as well as representation or readout limitations. Finally, rollout competence is measured in latent space against two references; without a memory-conditioned reward and executed policy, it cannot be converted into a control claim. These boundaries are why we report representation, retention, context, and rollout results separately rather than collapsing them into one aggregate score.

## 9. Conclusion

Persistent memory is not a longer window, a recurrent block, or a stable-looking rollout. In frozen SIGReg LeWM, GRU and diagonal SSM produce small paired improvements over no carrier on both tasks, but no tested carrier has a final-read interval clearly above four-way chance on both. At $H=56$, raw access returns while predictor output remains near chance. Learned dynamics nevertheless beat copy-last and use actions through $K=8$. Together these experiments distinguish representation, persistence, objective alignment, and dynamics under one official host. Joint training and memory-conditioned control remain future work.

APPENDIXMARKER

## Experimental Protocol and Data Contract

The appendix reports the complete validated grid behind the main-paper aggregates. It preserves semantic task names and separates primary endpoints from exploratory diagnostics. No appendix result expands the scope to memory-conditioned control.

\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Complete formal experiment matrix. Admission probes are computed before this 86-cell training grid.}
\label{tab:app-grid}
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xrrrr@{}}
\toprule
Study & Axes & Seeds & Epochs & Cells & Status \\
\midrule
Frozen carrier & 2 tasks $\times$ 5 carrier variants & 5 & 100 & 50 & Complete \\
Long context & 2 tasks $\times$ 4 context lengths & 3 & 60 & 24 & Complete \\
Learned rollout & 2 tasks $\times$ 2 objectives & 3 & 60 & 12 & Complete \\
\midrule
Total &  &  &  & 86 & 86/86 \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3.4pt}
\caption{Optimization and module-freezing contract. The rollout weight decay is a launcher setting rather than a serialized cell field.}
\label{tab:app-training}
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xrrrr@{}}
\toprule
Study & Frozen & Trainable & Batch & LR & Weight decay & Seeds \\
\midrule
Carrier & Encoder--projector, action encoder, predictor, output projection & Carrier only & 64 & 0.0003 & 0.00001 & 5 \\
Context & Encoder--projector & Action encoder, predictor, output projection & 256 & 0.0001 & 0.001 & 3 \\
Rollout & Encoder--projector & Action encoder, predictor, output projection & 64 & 0.0001 & 0.001* & 3 \\
\bottomrule
\end{tabularx}
\end{table}

All episodes contain 64 observations and five simulator actions per observation interval, flattened to one 10-D action block. Each task uses 1,200 training and 240 fixed validation episodes. The official host has a 192-D latent and context $H=3$. The long-context batch size was amended from 64 to 256 before any context metric completed; objectives, epochs, seeds, and endpoints were unchanged.

## Endpoint Legality and Statistical Contract

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Readout-relative claim contract. Each row identifies what the corresponding endpoint can and cannot establish.}
\label{tab:app-endpoints}
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}
\toprule
Question & Legal endpoint & Supports & Does not support \\
\midrule
Availability & Four frozen cue-window latents; pre-gap latents for the continuous target & Linear target availability & Complete information or downstream use \\
Final retention & $[z_{60},z_{61},z_{62},b_{63}]$; $b_{63}$ is read before $z_{63}$ & Conditional final carrier effect & End-to-end capacity or control \\
Trajectory diagnostic & Final raw context, mean post-cue prior, and $b_{63}$ & Aggregate-trajectory linear decodability & Decision-time retention \\
Raw context & Mean and last latent in $z[q-H:q]$ & Finite evidence access & Predictor exposure \\
Predictor & One contextual prediction from legal latents/actions & Output decodability & An isolated causal bottleneck \\
Rollout & Fixed anchor $t=24$; copy-last and shuffled-action references & Action-sensitive latent dynamics & Planning, reward, or control \\
\bottomrule
\end{tabularx}
\end{table}

The primary carrier endpoint excludes the decision observation and contains no temporal aggregation. The exploratory trajectory feature averages pre-observation priors from two frames after cue offset through $q=63$; it is intentionally a different, more permissive readout. All reported intervals are deterministic 20,000-draw percentile bootstraps over model/optimizer seeds. Carrier contrasts pair the same seed and pool tasks with equal weight; fixed data splits are never resampled.

## Carrier Implementations and Parameter Matching

\begin{table}[H]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Carrier capacity matching. Relative mismatch is measured against the 76,032-parameter fixed-trust carrier.}
\label{tab:app-carriers}
\begin{tabular}{lrrrrl}
\toprule
Carrier & State width & Parameters & $\Delta$ params & Rel. mismatch & Imagined update \\
\midrule
No carrier & 0 & 0 & -- & -- & No state \\
GRU & 74 & 75,924 & -108 & 0.0014 & Zero-latent, action-only \\
LSTM & 61 & 76,372 & +340 & 0.0045 & Zero-latent, action-only \\
Diagonal SSM & 192 & 76,032 & +0 & 0.0000 & Zero-latent, action-only \\
Fixed-trust & 192 & 76,032 & 0 & 0.0000 & Predict without correction \\
\bottomrule
\end{tabular}
\end{table}

All learned reads are zero-initialized and all variants use the same residual injection point. The GRU, LSTM, and SSM baselines emit a prior from the state before the current frame; fixed-trust memory emits the predictive mean before correction. The common frozen host is checked before and after every run.

## Complete Frozen-Carrier Results

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{2.7pt}
\renewcommand{\arraystretch}{1.08}
\caption{Complete frozen-carrier results. Final accuracy is the primary causal endpoint; trajectory accuracy is an exploratory temporally aggregated diagnostic. Arm entries are mean $\pm$ sample SD; paired contrasts retain 95\% seed-bootstrap intervals.}
\label{tab:app-frozen-full}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.19\linewidth}rrrr@{}}
\toprule
Task & Carrier & Final acc. & Trajectory acc. & $\Delta$ vs. none & Next-latent MSE \\
\midrule
Transient-marker recall & No carrier & $0.258\!\pm\!0.000$ & $0.258\!\pm\!0.000$ & -- & $0.2319\!\pm\!0.0000$ \\
 & Action-conditioned GRU & $0.274\!\pm\!0.008$ & $0.270\!\pm\!0.026$ & $+0.016\;[+0.009, +0.022]$ & $0.0795\!\pm\!0.0005$ \\
 & Action-conditioned LSTM & $0.272\!\pm\!0.018$ & $0.255\!\pm\!0.022$ & $+0.013\;[-0.001, +0.027]$ & $0.0850\!\pm\!0.0002$ \\
 & Diagonal SSM & $0.274\!\pm\!0.015$ & $0.506\!\pm\!0.027$ & $+0.016\;[+0.003, +0.026]$ & $0.0671\!\pm\!0.0001$ \\
 & Fixed-trust predict--correct & $0.278\!\pm\!0.012$ & $0.454\!\pm\!0.005$ & $+0.020\;[+0.009, +0.028]$ & $0.1018\!\pm\!0.0001$ \\
\midrule
Drifting-color recall & No carrier & $0.212\!\pm\!0.000$ & $0.212\!\pm\!0.000$ & -- & $0.3737\!\pm\!0.0000$ \\
 & Action-conditioned GRU & $0.227\!\pm\!0.015$ & $0.239\!\pm\!0.021$ & $+0.015\;[+0.003, +0.027]$ & $0.1569\!\pm\!0.0006$ \\
 & Action-conditioned LSTM & $0.226\!\pm\!0.018$ & $0.226\!\pm\!0.019$ & $+0.013\;[-0.002, +0.025]$ & $0.1667\!\pm\!0.0006$ \\
 & Diagonal SSM & $0.238\!\pm\!0.010$ & $0.421\!\pm\!0.046$ & $+0.026\;[+0.018, +0.033]$ & $0.1366\!\pm\!0.0001$ \\
 & Fixed-trust predict--correct & $0.202\!\pm\!0.004$ & $0.408\!\pm\!0.012$ & $-0.011\;[-0.014, -0.008]$ & $0.1790\!\pm\!0.0001$ \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Fixed-trust paired contrasts. Positive values favor fixed-trust; task-level wins are paired model seeds, while pooled wins count task--seed cells.}
\label{tab:app-frozen-contrasts}
\begin{tabular}{llrrr}
\toprule
Scope & Comparator & Difference [95\% CI] & Wins & Ties \\
\midrule
Transient-marker recall & Action-conditioned GRU & $+0.004\;[-0.012, +0.017]$ & 3/5 & 1 \\
 & Action-conditioned LSTM & $+0.007\;[-0.005, +0.016]$ & 3/5 & 1 \\
 & Diagonal SSM & $+0.004\;[-0.008, +0.022]$ & 2/5 & 1 \\
\addlinespace[1pt]
Drifting-color recall & Action-conditioned GRU & $-0.026\;[-0.039, -0.012]$ & 0/5 & 0 \\
 & Action-conditioned LSTM & $-0.024\;[-0.035, -0.008]$ & 1/5 & 0 \\
 & Diagonal SSM & $-0.037\;[-0.046, -0.028]$ & 0/5 & 0 \\
\addlinespace[1pt]
Equal-task pooled & Action-conditioned GRU & $-0.011\;[-0.021, -0.001]$ & 3/10 & 1 \\
 & Action-conditioned LSTM & $-0.009\;[-0.017, +0.001]$ & 4/10 & 1 \\
 & Diagonal SSM & $-0.016\;[-0.024, -0.007]$ & 2/10 & 1 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{4.0pt}
\caption{Seed-level final pre-observation accuracy. The no-carrier columns repeat by construction because that reference has no optimized state.}
\label{tab:app-frozen-seeds}
\begin{tabular}{lrrrrrr}
\toprule
Task & Seed & None & GRU & LSTM & SSM & Fixed-trust \\
\midrule
Transient-marker recall & 0 & 0.258 & 0.275 & 0.287 & 0.283 & 0.275 \\
 & 1 & 0.258 & 0.279 & 0.275 & 0.287 & 0.287 \\
 & 2 & 0.258 & 0.263 & 0.263 & 0.279 & 0.283 \\
 & 3 & 0.258 & 0.271 & 0.287 & 0.250 & 0.287 \\
 & 4 & 0.258 & 0.283 & 0.246 & 0.271 & 0.258 \\
\midrule
Drifting-color recall & 0 & 0.212 & 0.229 & 0.225 & 0.233 & 0.200 \\
 & 1 & 0.212 & 0.246 & 0.229 & 0.246 & 0.196 \\
 & 2 & 0.212 & 0.208 & 0.196 & 0.225 & 0.204 \\
 & 3 & 0.212 & 0.217 & 0.237 & 0.237 & 0.204 \\
 & 4 & 0.212 & 0.237 & 0.242 & 0.250 & 0.204 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/fig_a_appendix_probe.pdf}
\caption{Exploratory temporal aggregation versus the primary final causal endpoint. Differences are computed within seed. Positive values mean that the trajectory-average feature yields higher linear-probe accuracy than the primary final feature. Because their temporal support and feature maps differ, this is not evidence for decision-time retention and does not replace the main endpoint.}
\label{fig:app-probe}
\end{figure}

The exploratory diagnostic does not alter the pre-specified primary endpoint, and it produces a different carrier ordering. SSM and fixed-trust trajectory-average features are substantially more linearly decodable than their final features, whose causal reads remain near chance; GRU and LSTM show little such gap. This localizes decodable signal to the aggregate post-cue trajectory, not to any particular transient state. Because temporal support and feature maps differ, this remains a descriptive mechanism diagnostic rather than a formal paired endpoint.

## Complete Long-Context Controls

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3.2pt}
\caption{Complete long-context results. Raw-context accuracy is seed-independent; predictor and MSE entries are mean $\pm$ sample SD. Paired changes retain 95\% seed-bootstrap intervals. Cue coverage is measured on 240 validation episodes.}
\label{tab:app-context-full}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.21\linewidth}rrrrrY@{}}
\toprule
Task & $H$ & Cue frames (mean) & Episodes with cue & Raw acc. & Predictor acc. & Next-latent MSE \\
\midrule
Transient-marker recall & 3 & 0.0 & 0/240 & 0.229 & $0.261\!\pm\!0.006$ & $0.0524\!\pm\!0.0001$ \\
 & 16 & 0.0 & 0/240 & 0.250 & $0.253\!\pm\!0.002$ & $0.0132\!\pm\!0.0003$ \\
 & 32 & 0.0 & 0/240 & 0.263 & $0.251\!\pm\!0.002$ & $0.0049\!\pm\!0.0000$ \\
 & 56 & 4.9 & 240/240 & 0.771 & $0.253\!\pm\!0.005$ & $0.0065\!\pm\!0.0000$ \\
\midrule
Drifting-color recall & 3 & 0.0 & 0/240 & 0.171 & $0.208\!\pm\!0.004$ & $0.0837\!\pm\!0.0002$ \\
 & 16 & 0.0 & 0/240 & 0.179 & $0.192\!\pm\!0.008$ & $0.0587\!\pm\!0.0001$ \\
 & 32 & 0.0 & 0/240 & 0.196 & $0.183\!\pm\!0.007$ & $0.0525\!\pm\!0.0002$ \\
 & 56 & 5.0 & 240/240 & 0.825 & $0.190\!\pm\!0.002$ & $0.0662\!\pm\!0.0002$ \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Paired long-context changes relative to $H=3$. Negative MSE differences indicate better local prediction.}
\label{tab:app-context-contrasts}
\begin{tabular}{lrrr}
\toprule
Task & Comparison & $\Delta$ predictor accuracy & $\Delta$ next-latent MSE \\
\midrule
Transient-marker recall & $H=16$ minus $H=3$ & $-0.008\;[-0.013, -0.004]$ & $-0.0393\;[-0.0394, -0.0391]$ \\
 & $H=32$ minus $H=3$ & $-0.010\;[-0.013, -0.004]$ & $-0.0475\;[-0.0476, -0.0474]$ \\
 & $H=56$ minus $H=3$ & $-0.008\;[-0.013, -0.004]$ & $-0.0460\;[-0.0461, -0.0458]$ \\
\midrule
Drifting-color recall & $H=16$ minus $H=3$ & $-0.017\;[-0.021, -0.008]$ & $-0.0250\;[-0.0253, -0.0247]$ \\
 & $H=32$ minus $H=3$ & $-0.025\;[-0.033, -0.017]$ & $-0.0312\;[-0.0312, -0.0312]$ \\
 & $H=56$ minus $H=3$ & $-0.018\;[-0.021, -0.017]$ & $-0.0175\;[-0.0178, -0.0172]$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/fig_a_appendix_context.pdf}
\caption{Local prediction loss versus delayed semantic exposure. Each point is a separately trained context model; pale paths connect the same model seed across context lengths. The cue first enters legal context at $H=56$, but predictor semantic accuracy does not follow the raw-context jump. Lines are descriptive and are not fitted trends.}
\label{fig:app-context}
\end{figure}

Both tasks achieve their lowest local next-latent error at $H=32$, where no validation window contains a cue frame. At $H=56$, every validation episode contains cue evidence and raw readout rises sharply, yet predictor semantic accuracy remains near chance. The change from $H=32$ to $H=56$ therefore improves access while worsening the local MSE objective. Since models and readouts are re-estimated for each $H$, this supports a descriptive objective--semantic dissociation, not a causal decomposition.

## Complete Learned-Rollout Analysis

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3.0pt}
\caption{Complete learned-rollout primary diagnostics for Transient-marker recall. Entries are mean [95\% seed-bootstrap CI].}
\label{tab:app-rollout-primary-marker}
\begin{tabular}{lrrrrr}
\toprule
Objective & $K$ & Normalized MSE & Copy-last MSE & Model/copy ratio & True-action advantage \\
\midrule
One-step & 1 & $0.008\;[0.008, 0.009]$ & $0.142\;[0.142, 0.142]$ & $0.057\;[0.054, 0.061]$ & $0.224\;[0.201, 0.251]$ \\
 & 2 & $0.013\;[0.012, 0.014]$ & $0.334\;[0.334, 0.334]$ & $0.037\;[0.034, 0.042]$ & $0.512\;[0.497, 0.533]$ \\
 & 4 & $0.021\;[0.019, 0.026]$ & $0.621\;[0.621, 0.621]$ & $0.034\;[0.030, 0.041]$ & $0.891\;[0.878, 0.904]$ \\
 & 8 & $0.040\;[0.034, 0.051]$ & $0.933\;[0.933, 0.933]$ & $0.043\;[0.036, 0.054]$ & $1.258\;[1.229, 1.288]$ \\
 & 16 & $0.100\;[0.083, 0.132]$ & $1.377\;[1.377, 1.377]$ & $0.072\;[0.060, 0.096]$ & $1.545\;[1.490, 1.599]$ \\
\midrule
Eight-step overshooting & 1 & $0.015\;[0.014, 0.016]$ & $0.142\;[0.142, 0.142]$ & $0.106\;[0.101, 0.114]$ & $0.217\;[0.197, 0.244]$ \\
 & 2 & $0.020\;[0.019, 0.021]$ & $0.334\;[0.334, 0.334]$ & $0.060\;[0.057, 0.064]$ & $0.493\;[0.471, 0.519]$ \\
 & 4 & $0.027\;[0.025, 0.029]$ & $0.621\;[0.621, 0.621]$ & $0.044\;[0.041, 0.047]$ & $0.866\;[0.854, 0.882]$ \\
 & 8 & $0.039\;[0.037, 0.041]$ & $0.933\;[0.933, 0.933]$ & $0.042\;[0.040, 0.044]$ & $1.226\;[1.215, 1.234]$ \\
 & 16 & $0.073\;[0.066, 0.086]$ & $1.377\;[1.377, 1.377]$ & $0.053\;[0.048, 0.062]$ & $1.532\;[1.512, 1.545]$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3.0pt}
\caption{Complete learned-rollout primary diagnostics for Drifting-color recall. Entries are mean [95\% seed-bootstrap CI].}
\label{tab:app-rollout-primary-color}
\begin{tabular}{lrrrrr}
\toprule
Objective & $K$ & Normalized MSE & Copy-last MSE & Model/copy ratio & True-action advantage \\
\midrule
One-step & 1 & $0.066\;[0.065, 0.066]$ & $0.140\;[0.140, 0.140]$ & $0.469\;[0.466, 0.474]$ & $0.111\;[0.107, 0.116]$ \\
 & 2 & $0.110\;[0.108, 0.110]$ & $0.266\;[0.266, 0.266]$ & $0.411\;[0.407, 0.414]$ & $0.234\;[0.217, 0.243]$ \\
 & 4 & $0.195\;[0.194, 0.199]$ & $0.539\;[0.539, 0.539]$ & $0.362\;[0.359, 0.369]$ & $0.476\;[0.441, 0.496]$ \\
 & 8 & $0.301\;[0.295, 0.311]$ & $0.916\;[0.916, 0.916]$ & $0.329\;[0.322, 0.340]$ & $0.766\;[0.743, 0.803]$ \\
 & 16 & $0.370\;[0.352, 0.398]$ & $1.134\;[1.134, 1.134]$ & $0.326\;[0.310, 0.351]$ & $0.900\;[0.863, 0.919]$ \\
\midrule
Eight-step overshooting & 1 & $0.083\;[0.082, 0.084]$ & $0.140\;[0.140, 0.140]$ & $0.594\;[0.587, 0.601]$ & $0.110\;[0.107, 0.114]$ \\
 & 2 & $0.124\;[0.123, 0.124]$ & $0.266\;[0.266, 0.266]$ & $0.464\;[0.462, 0.467]$ & $0.230\;[0.219, 0.238]$ \\
 & 4 & $0.181\;[0.180, 0.182]$ & $0.539\;[0.539, 0.539]$ & $0.335\;[0.334, 0.337]$ & $0.470\;[0.444, 0.489]$ \\
 & 8 & $0.220\;[0.217, 0.223]$ & $0.916\;[0.916, 0.916]$ & $0.241\;[0.237, 0.244]$ & $0.770\;[0.751, 0.805]$ \\
 & 16 & $0.248\;[0.240, 0.254]$ & $1.134\;[1.134, 1.134]$ & $0.219\;[0.211, 0.224]$ & $0.937\;[0.900, 0.968]$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{2.8pt}
\caption{Secondary learned-rollout diagnostics. Shuffled-action MSE tests action dependence; pose MAE is a linear physical-state readout. Entries are mean $\pm$ sample SD; the plotted rank ratio is computed per seed before aggregation.}
\label{tab:app-rollout-secondary}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.18\linewidth}rrrrY@{}}
\toprule
Task & Objective & $K$ & Shuffled MSE & Pose MAE & Predicted rank & Target rank \\
\midrule
Transient-marker recall & One-step & 1 & $0.232\!\pm\!0.024$ & $0.358\!\pm\!0.003$ & $64.87\!\pm\!0.09$ & $64.58\!\pm\!0.00$ \\
 &  & 2 & $0.525\!\pm\!0.018$ & $0.361\!\pm\!0.002$ & $65.78\!\pm\!0.15$ & $65.68\!\pm\!0.00$ \\
 &  & 4 & $0.912\!\pm\!0.010$ & $0.359\!\pm\!0.008$ & $65.70\!\pm\!0.21$ & $66.42\!\pm\!0.00$ \\
 &  & 8 & $1.298\!\pm\!0.031$ & $0.353\!\pm\!0.003$ & $63.93\!\pm\!0.48$ & $64.91\!\pm\!0.00$ \\
 &  & 16 & $1.645\!\pm\!0.034$ & $0.340\!\pm\!0.010$ & $64.47\!\pm\!0.88$ & $65.94\!\pm\!0.00$ \\
\addlinespace[1pt]
 & Eight-step overshooting & 1 & $0.232\!\pm\!0.024$ & $0.397\!\pm\!0.032$ & $64.13\!\pm\!0.19$ & $64.58\!\pm\!0.00$ \\
 &  & 2 & $0.513\!\pm\!0.023$ & $0.401\!\pm\!0.029$ & $64.91\!\pm\!0.33$ & $65.68\!\pm\!0.00$ \\
 &  & 4 & $0.893\!\pm\!0.013$ & $0.396\!\pm\!0.029$ & $65.03\!\pm\!0.54$ & $66.42\!\pm\!0.00$ \\
 &  & 8 & $1.265\!\pm\!0.012$ & $0.383\!\pm\!0.028$ & $63.99\!\pm\!0.49$ & $64.91\!\pm\!0.00$ \\
 &  & 16 & $1.605\!\pm\!0.007$ & $0.377\!\pm\!0.026$ & $63.98\!\pm\!0.59$ & $65.94\!\pm\!0.00$ \\
\midrule
Drifting-color recall & One-step & 1 & $0.177\!\pm\!0.005$ & $0.502\!\pm\!0.003$ & $37.92\!\pm\!0.15$ & $37.32\!\pm\!0.00$ \\
 &  & 2 & $0.343\!\pm\!0.015$ & $0.515\!\pm\!0.002$ & $37.77\!\pm\!0.34$ & $37.53\!\pm\!0.00$ \\
 &  & 4 & $0.671\!\pm\!0.032$ & $0.510\!\pm\!0.001$ & $37.66\!\pm\!0.72$ & $38.01\!\pm\!0.00$ \\
 &  & 8 & $1.068\!\pm\!0.041$ & $0.542\!\pm\!0.005$ & $37.26\!\pm\!1.37$ & $37.97\!\pm\!0.00$ \\
 &  & 16 & $1.270\!\pm\!0.008$ & $0.579\!\pm\!0.006$ & $37.24\!\pm\!2.07$ & $38.84\!\pm\!0.00$ \\
\addlinespace[1pt]
 & Eight-step overshooting & 1 & $0.193\!\pm\!0.005$ & $0.567\!\pm\!0.027$ & $38.77\!\pm\!0.07$ & $37.32\!\pm\!0.00$ \\
 &  & 2 & $0.354\!\pm\!0.010$ & $0.548\!\pm\!0.018$ & $38.01\!\pm\!0.15$ & $37.53\!\pm\!0.00$ \\
 &  & 4 & $0.651\!\pm\!0.024$ & $0.526\!\pm\!0.020$ & $37.80\!\pm\!0.32$ & $38.01\!\pm\!0.00$ \\
 &  & 8 & $0.991\!\pm\!0.031$ & $0.529\!\pm\!0.013$ & $37.50\!\pm\!0.41$ & $37.97\!\pm\!0.00$ \\
 &  & 16 & $1.185\!\pm\!0.036$ & $0.516\!\pm\!0.013$ & $37.84\!\pm\!0.72$ & $38.84\!\pm\!0.00$ \\
\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figures/fig_a_appendix_rollout.pdf}
\caption{Secondary rollout diagnostics. Pose MAE is obtained from a linear state readout; effective rank is normalized by the paired target rank within seed. Ratios near one show no obvious global covariance-rank collapse in this diagnostic, but do not establish cue retention or physical correctness. The rank panel uses a zoomed vertical scale.}
\label{fig:app-rollout}
\end{figure}

\begin{table}[H]
\centering\scriptsize
\setlength{\tabcolsep}{3.2pt}
\caption{Paired eight-step-overshooting minus one-step contrasts. Negative MSE and pose differences are improvements; positive action-advantage differences indicate stronger action dependence.}
\label{tab:app-rollout-contrasts}
\begin{tabular}{lrrrr}
\toprule
Task & $K$ & $\Delta$ normalized MSE & $\Delta$ pose MAE & $\Delta$ action advantage \\
\midrule
Transient-marker recall & 1 & $+0.0070\;[+0.0057, +0.0086]$ & $+0.0390\;[+0.0218, +0.0728]$ & $-0.0068\;[-0.0095, -0.0045]$ \\
 & 2 & $+0.0075\;[+0.0054, +0.0099]$ & $+0.0406\;[+0.0209, +0.0715]$ & $-0.0191\;[-0.0258, -0.0130]$ \\
 & 4 & $+0.0059\;[+0.0017, +0.0101]$ & $+0.0368\;[+0.0230, +0.0617]$ & $-0.0249\;[-0.0426, -0.0078]$ \\
 & 8 & $-0.0006\;[-0.0098, +0.0050]$ & $+0.0296\;[+0.0120, +0.0598]$ & $-0.0318\;[-0.0540, -0.0134]$ \\
 & 16 & $-0.0267\;[-0.0463, -0.0154]$ & $+0.0375\;[+0.0191, +0.0664]$ & $-0.0133\;[-0.0611, +0.0222]$ \\
\midrule
Drifting-color recall & 1 & $+0.0174\;[+0.0167, +0.0178]$ & $+0.0654\;[+0.0411, +0.0947]$ & $-0.0016\;[-0.0027, -0.0005]$ \\
 & 2 & $+0.0142\;[+0.0129, +0.0151]$ & $+0.0325\;[+0.0177, +0.0494]$ & $-0.0037\;[-0.0068, +0.0011]$ \\
 & 4 & $-0.0146\;[-0.0170, -0.0132]$ & $+0.0160\;[-0.0061, +0.0322]$ & $-0.0055\;[-0.0123, +0.0027]$ \\
 & 8 & $-0.0810\;[-0.0903, -0.0712]$ & $-0.0140\;[-0.0259, -0.0078]$ & $+0.0042\;[-0.0018, +0.0124]$ \\
 & 16 & $-0.1220\;[-0.1482, -0.0979]$ & $-0.0634\;[-0.0720, -0.0486]$ & $+0.0372\;[+0.0254, +0.0493]$ \\
\bottomrule
\end{tabular}
\end{table}

Overshooting exhibits a horizon- and task-dependent trade-off. It increases short-horizon latent error on both tasks. For drifting-color recall it improves latent MSE from $K=4$ onward and improves pose MAE at the longest horizons. For transient-marker recall, the $K=16$ latent MSE improves while pose MAE worsens. Consequently, lower latent rollout error is not a uniform proxy for physical readout quality. Across all conditions, predicted/target effective-rank ratios stay close to one; this diagnostic shows no evidence of an obvious global rank collapse, but says nothing about the earlier cue semantics.

## Reproducibility Ledger and Claim Boundary

\begin{table}[H]
\centering\small
\caption{Publication-artifact validation ledger. Full hashes and all 86 source-metric records are retained in the machine-readable manifest.}
\label{tab:app-ledger}
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xl@{}}
\toprule
Check & Validated condition & Status \\
\midrule
Grid & 86 of 86 metric cells present & Pass \\
Official host & Released source, commit, and weight SHA-256 match & Pass \\
Frozen host & Before/after state digest identical in every carrier run & Pass \\
Carrier checkpoints & Embedded metrics, state digest, and cache receipts match & Pass \\
No carrier & Repeated seed records are exactly deterministic & Pass \\
Aggregation & Seed-level statistics and 20,000-draw intervals regenerated & Pass \\
\bottomrule
\end{tabularx}
\end{table}

The study does not establish that fixed-trust memory is superior to GRU or SSM, that the continuous target is absent from the representation, or that longer context causally identifies an objective bottleneck. The rollout prerequisite contains no reward, CEM return, or executed policy result; passing it cannot be promoted to planning or control. Intervals cover optimization/model seeds on fixed splits, not new-task or new-environment uncertainty. These boundaries apply equally to the main paper and the additional appendix diagnostics.

The official host source, commit, complete weight hash, configuration record, cache manifests, and source-metric hashes are distributed with the artifact. The PDF prints the scientific contract and aggregate results; the machine-readable ledger remains the authority for exact provenance.

