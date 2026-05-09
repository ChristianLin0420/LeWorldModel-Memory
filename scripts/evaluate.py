"""
Evaluation script for LeWorldModel.
Runs planning evaluation on an environment and reports success rate.
"""

import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.leworldmodel import LeWorldModel
from lewm.eval.probing import planning_evaluation
from lewm.envs.two_room import TwoRoomEnv


def main():
    parser = argparse.ArgumentParser(description='Evaluate LeWorldModel')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--env', type=str, default='two_room', choices=['two_room'])
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--planning-horizon', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--num-elites', type=int, default=30)
    parser.add_argument('--num-iterations', type=int, default=30)
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Evaluating LeWorldModel on {args.env}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Episodes: {args.num_episodes}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get('args', {})

    # Create model
    model = LeWorldModel(
        img_size=model_args.get('img_size', args.img_size),
        patch_size=model_args.get('patch_size', 14 if args.img_size >= 224 else 8),
        embed_dim=model_args.get('embed_dim', 192),
        action_dim=model_args.get('action_dim', 2),
        encoder_layers=model_args.get('encoder_layers', 12),
        encoder_heads=model_args.get('encoder_heads', 3),
        predictor_layers=model_args.get('predictor_layers', 6),
        predictor_heads=model_args.get('predictor_heads', 16),
        history_len=model_args.get('history_len', 3),
        dropout=model_args.get('dropout', 0.1),
        sigreg_lambda=model_args.get('sigreg_lambda', 0.1),
        sigreg_projections=model_args.get('sigreg_projections', 1024),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Trained for {ckpt.get('epoch', '?')} epochs")

    # Create environment
    if args.env == 'two_room':
        env = TwoRoomEnv(img_size=args.img_size, max_steps=args.max_steps)
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Run evaluation
    results = planning_evaluation(
        model=model,
        env=env,
        num_episodes=args.num_episodes,
        planning_horizon=args.planning_horizon,
        max_steps=args.max_steps,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        device=device,
    )

    print("\n" + "=" * 40)
    print("Results:")
    print(f"  Success Rate: {results['success_rate']:.1%}")
    print(f"  Mean Steps:   {results['mean_steps']:.1f}")
    print(f"  Mean Reward:  {results['mean_reward']:.4f}")
    print("=" * 40)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
