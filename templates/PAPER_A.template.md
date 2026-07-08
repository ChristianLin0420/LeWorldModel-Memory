# Beyond the Context Window: A Frozen-Host Audit of Persistent Memory in SIGReg LeWM

## Abstract

Latent world models can predict local dynamics while forgetting evidence after it leaves their observation window. Using the released SIGReg LeWorldModel (LeWM) Reacher checkpoint, we audit representation availability, causal retention, finite-context access, and rollout competence. All released components remain frozen while parameter-matched GRU, LSTM, diagonal state-space, and fixed-trust predict--correct carriers are optimized through the next-latent loss. Transient-marker and drifting-color recall pass the representation criterion; occluded-target prediction reaches only $R^2={{OCCLUDED_AVAILABILITY_R2}}$ and is excluded at the pre-specified $.300$ threshold. Across the admitted tasks, fixed-trust trails the GRU by {{POOLED_FIXED_TRUST_GRU_DIFFERENCE}} accuracy (95\% CI $[{{POOLED_FIXED_TRUST_GRU_CI_LOW}},{{POOLED_FIXED_TRUST_GRU_CI_HIGH}}]$). {{ABSTRACT_CONTEXT_AND_ROLLOUT_FINDING}} These results isolate memory effects under a common official host; memory-conditioned control is outside their scope.

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
\caption{Memory tasks and frozen-representation admission. Categorical rows show the cue, class-independent delay, and last legal pre-decision frame; their evidence age is 43--53 frames. The continuous row shows the last pre-gap observation, a 16--20-frame frozen interval, and an evaluation-only outcome after the cutoff. Four cue-window latents yield categorical accuracy above $.750$; four pre-gap latents yield only $R^2={{OCCLUDED_AVAILABILITY_R2}}$ for the continuous target, below $.300$.}
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

Figure \ref{fig:protocol} admits only the two categorical tasks: transient-marker and drifting-color accuracy are {{MARKER_AVAILABILITY_ACC}} and {{DRIFTING_AVAILABILITY_ACC}}. Occluded-target prediction remains an availability diagnostic; its failure does not establish absence of the current pre-gap state.

### 5.2 Frozen-host carrier swap

Table \ref{tab:frozen-results} and Figure \ref{fig:memory-results}a report the carrier comparison. On transient-marker recall, no carrier obtains {{MARKER_NONE_ACC_WITH_CI}}, fixed-trust {{MARKER_FIXED_TRUST_ACC_WITH_CI}}, and GRU {{MARKER_GRU_ACC_WITH_CI}}. On drifting-color recall, they obtain {{DRIFTING_NONE_ACC_WITH_CI}}, {{DRIFTING_FIXED_TRUST_ACC_WITH_CI}}, and {{DRIFTING_GRU_ACC_WITH_CI}}, respectively.

The pre-specified primary contrast is fixed-trust memory minus the parameter-matched GRU, paired by seed and weighted equally across tasks. Its mean is {{POOLED_FIXED_TRUST_GRU_DIFFERENCE}} with 95\% interval $[{{POOLED_FIXED_TRUST_GRU_CI_LOW}},{{POOLED_FIXED_TRUST_GRU_CI_HIGH}}]$ and {{POOLED_FIXED_TRUST_GRU_WINS}} positive seed--task pairs. The corresponding contrasts against LSTM and the diagonal state-space carrier are {{POOLED_FIXED_TRUST_LSTM_DIFFERENCE_WITH_CI}} and {{POOLED_FIXED_TRUST_SSM_DIFFERENCE_WITH_CI}}. {{FROZEN_SWAP_INTERPRETATION}}

Figure \ref{fig:memory-results}a makes the aggregation transparent. Fixed-trust is numerically positive against all learned comparators on transient-marker recall, but every task-specific interval crosses zero. On drifting-color recall all three contrasts are negative. Equal task weighting therefore prevents the marker task from hiding the color failure. {{MAIN_NO_CARRIER_ANALYSIS}}

Next-latent error is reported beside accuracy because local prediction and remote retention need not agree. Host, data, context, loss, optimizer, and readout remain fixed while only recurrent state changes.

The loss ranking sharpens this point. The diagonal state-space carrier gives the lowest pooled next-latent MSE, followed by the GRU, LSTM, fixed-trust memory, and no carrier, yet all equal-task accuracies remain close to $.250$ chance. Thus reduced local loss does not establish persistent target information. {{MAIN_TRAJECTORY_ANALYSIS}}

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
No persistent carrier & 0 & {{MARKER_NONE_ACC_WITH_CI}} & {{DRIFTING_NONE_ACC_WITH_CI}} & {{NONE_MEAN_ACC_WITH_CI}} & {{NONE_MEAN_NEXT_LATENT_MSE}} \\
\rowcolor{TableGray}
Action-conditioned GRU & 75,924 & {{MARKER_GRU_ACC_WITH_CI}} & {{DRIFTING_GRU_ACC_WITH_CI}} & {{GRU_MEAN_ACC_WITH_CI}} & {{GRU_MEAN_NEXT_LATENT_MSE}} \\
Action-conditioned LSTM & 76,372 & {{MARKER_LSTM_ACC_WITH_CI}} & {{DRIFTING_LSTM_ACC_WITH_CI}} & {{LSTM_MEAN_ACC_WITH_CI}} & {{LSTM_MEAN_NEXT_LATENT_MSE}} \\
\rowcolor{TableGray}
Diagonal state-space carrier & 76,032 & {{MARKER_SSM_ACC_WITH_CI}} & {{DRIFTING_SSM_ACC_WITH_CI}} & {{SSM_MEAN_ACC_WITH_CI}} & {{SSM_MEAN_NEXT_LATENT_MSE}} \\
\addlinespace[1pt]
Fixed-trust predict--correct (ours) & 76,032 & {{MARKER_FIXED_TRUST_ACC_WITH_CI}} & {{DRIFTING_FIXED_TRUST_ACC_WITH_CI}} & {{FIXED_TRUST_MEAN_ACC_WITH_CI}} & {{FIXED_TRUST_MEAN_NEXT_LATENT_MSE}} \\
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

Figure \ref{fig:memory-results}b compares raw access with a predictor trained at each context. For transient-marker recall, raw accuracy changes from {{MARKER_RAW_CONTEXT_H3_ACC}} at $H=3$ to {{MARKER_RAW_CONTEXT_H56_ACC}} at $H=56$, while predictor accuracy changes from {{MARKER_PREDICTOR_H3_ACC_WITH_CI}} to {{MARKER_PREDICTOR_H56_ACC_WITH_CI}}. For drifting-color recall, raw accuracy moves from {{DRIFTING_RAW_CONTEXT_H3_ACC}} to {{DRIFTING_RAW_CONTEXT_H56_ACC}}, versus {{DRIFTING_PREDICTOR_H3_ACC_WITH_CI}} to {{DRIFTING_PREDICTOR_H56_ACC_WITH_CI}} for the predictor.

The raw curve asks whether the event is readable once it re-enters legal context; the predictor curve asks whether next-latent training exposes it. {{LONG_CONTEXT_INTERPRETATION}} Because the predictor and readout are re-estimated at each $H$, this is a capability control rather than an isolated carrier effect. The pattern is consistent with, but does not prove, an objective or predictor bottleneck.

Coverage receipts explain the discontinuity. Among the tested context lengths, at $H\in\{3,16,32\}$ no validation window contains a cue frame; at $H=56$, every validation episode contains some cue evidence, averaging about five cue frames. {{MAIN_CONTEXT_MSE_ANALYSIS}}

## 6. Learned Rollout Competence

Persistent memory and imagination solve different temporal problems. We evaluate learned dynamics separately before any memory-conditioned control claim.

Figure \ref{fig:rollouts}a reports paired latent-MSE ratios to copy-last; values below one improve on repeating anchor $t=24$. Figure \ref{fig:rollouts}b reports variance-normalized shuffled-action MSE minus true-action MSE; positive values indicate useful action conditioning.

At horizon eight, the one-step objective obtains MSE/copy-last ratios of {{MARKER_ONE_STEP_H8_MSE_RATIO}} on Transient-marker recall and {{DRIFTING_ONE_STEP_H8_MSE_RATIO}} on drifting-color recall. Eight-step overshooting obtains {{MARKER_OVERSHOOT_H8_MSE_RATIO}} and {{DRIFTING_OVERSHOOT_H8_MSE_RATIO}}, respectively. True-action advantages at the same horizon are {{MARKER_ONE_STEP_H8_ACTION_ADVANTAGE}} and {{DRIFTING_ONE_STEP_H8_ACTION_ADVANTAGE}} for one-step training, versus {{MARKER_OVERSHOOT_H8_ACTION_ADVANTAGE}} and {{DRIFTING_OVERSHOOT_H8_ACTION_ADVANTAGE}} for overshooting.

The full horizon profile is more informative than a single endpoint. All error ratios stay below one through diagnostic $K=16$, and action advantage grows rather than collapsing toward zero. Overshooting sacrifices short-horizon accuracy, is tied with one-step training at $K=8$ for transient-marker recall, and helps drifting-color recall at longer horizons. {{MAIN_ROLLOUT_TRADEOFF}}

{{ROLLOUT_GATE_INTERPRETATION}}

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
Cue-time frozen representation & {{MARKER_AVAILABILITY_ACC}} & {{DRIFTING_AVAILABILITY_ACC}} & $R^2={{OCCLUDED_AVAILABILITY_R2}}$ & Admission \\
\rowcolor{TableGray}
Final fixed-trust read & {{MARKER_FIXED_TRUST_ACC_WITH_CI}} & {{DRIFTING_FIXED_TRUST_ACC_WITH_CI}} & Excluded & Fixed-trust endpoint \\
Raw context at $H=56$ & {{MARKER_RAW_CONTEXT_H56_ACC}} & {{DRIFTING_RAW_CONTEXT_H56_ACC}} & -- & Finite access \\
\rowcolor{TableGray}
Predictor output at $H=56$ & {{MARKER_PREDICTOR_H56_ACC_WITH_CI}} & {{DRIFTING_PREDICTOR_H56_ACC_WITH_CI}} & -- & Learned exposure \\
Rollout gate (one-step / overshoot) & {{MARKER_ONE_STEP_GATE_PASSES}} / {{MARKER_OVERSHOOT_GATE_PASSES}} & {{DRIFTING_ONE_STEP_GATE_PASSES}} / {{DRIFTING_OVERSHOOT_GATE_PASSES}} & -- & Dynamics only \\
\bottomrule
\end{tabularx}
\end{table}

The experiments give three counterexamples to common memory proxies. First, low next-latent loss does not imply final semantic retention. Once a class stops affecting future latents, the next-latent objective provides no predictive pressure to preserve it. {{PERSISTENCE_LOCALIZATION}} The trajectory-average diagnostic further separates aggregate post-cue decodability from a stable final read.

Second, placing a cue inside legal context does not imply that the trained predictor exposes it. {{CONTEXT_CARRIER_LOCALIZATION}} The $H=56$ raw gain and near-chance predictor output separate finite access from learned compression, while the feature-map mismatch prevents a causal bottleneck claim.

Third, rollout competence and memory retention use different evidence windows. {{DYNAMICS_LOCALIZATION}}

{{MAIN_TASK_HETEROGENEITY}}

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

{{APPENDIX_BODY}}
