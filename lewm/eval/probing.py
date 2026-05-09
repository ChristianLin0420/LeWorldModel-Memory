"""
Evaluation utilities for LeWorldModel.
Includes: latent probing, violation-of-expectation, planning evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr


@torch.no_grad()
def extract_latents(
    model: nn.Module,
    observations: torch.Tensor,
    batch_size: int = 256,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Extract latent embeddings from the encoder.

    Args:
        model: LeWorldModel with encoder
        observations: (N, C, H, W) tensor of observations
        batch_size: batch size for encoding
        device: torch device

    Returns:
        latents: (N, D) numpy array of latent embeddings
    """
    model.eval()
    latents = []
    for i in range(0, len(observations), batch_size):
        batch = observations[i:i+batch_size].to(device)
        z = model.encode(batch)
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def probe_latent_space(
    latents: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """
    Probe latent space with linear and MLP probes.

    Args:
        latents: (N, D) latent embeddings
        targets: (N, K) target physical quantities
        train_ratio: fraction of data for training

    Returns:
        results: dict with 'linear' and 'mlp' keys, each containing
                 'mse' and 'r' (Pearson correlation) arrays
    """
    N = len(latents)
    n_train = int(N * train_ratio)

    # Shuffle
    perm = np.random.permutation(N)
    latents = latents[perm]
    targets = targets[perm]

    train_latents = latents[:n_train]
    test_latents = latents[n_train:]
    train_targets = targets[:n_train]
    test_targets = targets[n_train:]

    results = {}

    # Linear probe
    linear = LinearRegression()
    linear.fit(train_latents, train_targets)
    pred_linear = linear.predict(test_latents)

    mse_linear = np.mean((pred_linear - test_targets) ** 2, axis=0)
    r_linear = np.array([
        pearsonr(pred_linear[:, i], test_targets[:, i])[0]
        for i in range(test_targets.shape[1])
    ])

    results['linear'] = {'mse': mse_linear, 'r': r_linear}

    # MLP probe
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(train_latents, train_targets)
    pred_mlp = mlp.predict(test_latents)

    mse_mlp = np.mean((pred_mlp - test_targets) ** 2, axis=0)
    r_mlp = np.array([
        pearsonr(pred_mlp[:, i], test_targets[:, i])[0]
        for i in range(test_targets.shape[1])
    ])

    results['mlp'] = {'mse': mse_mlp, 'r': r_mlp}

    return results


@torch.no_grad()
def violation_of_expectation(
    model: nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    perturbation_frame: int,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """
    Compute surprise (prediction error) over a trajectory.
    Higher surprise at the perturbation frame indicates the model
    detected the violation.

    Args:
        model: LeWorldModel
        observations: (T, C, H, W) observation sequence
        actions: (T-1, A) action sequence
        perturbation_frame: frame index where perturbation occurs
        device: torch device

    Returns:
        surprise: (T-1,) array of prediction errors (MSE) at each step
    """
    model.eval()
    T = len(observations)
    surprise = np.zeros(T - 1)

    obs = observations.to(device)
    act = actions.to(device)

    # Encode all observations
    z_all = model.encode(obs.unsqueeze(0)).squeeze(0)  # (T, D)

    # Compute prediction error at each step
    for t in range(T - 1):
        z_t = z_all[t:t+1].unsqueeze(0)  # (1, 1, D)
        a_t = act[t:t+1].unsqueeze(0)  # (1, 1, A)
        z_pred = model.predictor(z_t, a_t)  # (1, 1, D)
        z_actual = z_all[t+1:t+2].unsqueeze(0)  # (1, 1, D)

        error = F.mse_loss(z_pred, z_actual).item()
        surprise[t] = error

    return surprise


def planning_evaluation(
    model: nn.Module,
    env,
    num_episodes: int = 50,
    planning_horizon: int = 5,
    max_steps: int = 50,
    num_samples: int = 300,
    num_elites: int = 30,
    num_iterations: int = 30,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, float]:
    """
    Evaluate planning performance in an environment.

    Args:
        model: LeWorldModel
        env: Gymnasium environment
        num_episodes: number of evaluation episodes
        planning_horizon: planning horizon for CEM
        max_steps: maximum steps per episode
        num_samples: CEM samples per iteration
        num_elites: CEM elite count
        num_iterations: CEM optimization iterations
        device: torch device

    Returns:
        results: dict with 'success_rate', 'mean_steps', 'mean_reward'
    """
    successes = []
    step_counts = []
    rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        goal_obs = info.get('goal_obs', obs)  # Assume goal is in info
        total_reward = 0.0

        for step in range(max_steps):
            # Plan action
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            goal_tensor = torch.from_numpy(goal_obs).float().unsqueeze(0).to(device)

            # Get action from model's plan method
            # Note: env action space bounds needed
            action_space = env.action_space
            low = torch.from_numpy(action_space.low).float()
            high = torch.from_numpy(action_space.high).float()

            action = model.plan(
                obs_init=obs_tensor,
                obs_goal=goal_tensor,
                horizon=planning_horizon,
                num_samples=num_samples,
                num_elites=num_elites,
                num_iterations=num_iterations,
                action_bounds=(low, high),
            )

            action_np = action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

            if terminated or truncated:
                break

        success = info.get('success', terminated)
        successes.append(float(success))
        step_counts.append(step + 1)
        rewards.append(total_reward)

    return {
        'success_rate': np.mean(successes),
        'mean_steps': np.mean(step_counts),
        'mean_reward': np.mean(rewards),
    }
