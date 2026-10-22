"""
Open-loop evaluation of the Motus policy on recorded RoboTwin episodes (no simulator needed).

For each (episode, condition frame t) the policy sees the real observation at t and predicts a 16-step action chunk
and 8 future frames. They are compared with the recorded trajectory, using the same temporal layout as training
(action i at t + (i+1)*global_downsample_rate, video frame j at action (j+1)*video_action_freq_ratio - 1),
against trivial baselines: "hold current joint state" for actions, "repeat the condition frame" for video.

Outputs in --out_dir: per-sample frames_*.png / actions_*.png / rollout_*.gif, results.json, summary.md

Usage:
    MOTUS_T5_CACHE=t5_cache.pt python open_loop_eval.py --ckpt ... --wan_path ... --vlm_path ... \
        --task_dir /path/to/robotwin_raw/dataset/click_alarmclock --out_dir results/click_alarmclock
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from deploy_policy import get_model, reset_model
from smoke_test import load_obs
from utils.image_utils import resize_with_padding

JOINT_NAMES = [f'L{i}' for i in range(6)] + ['L_grip'] + [f'R{i}' for i in range(6)] + ['R_grip']


def compose_view(obs, size):
    """Three-view image exactly as MotusPolicy.update_obs builds it, as float [0,1] HxWx3."""
    cams = obs['observation']
    bottom = np.concatenate([cv2.resize(cams['left_camera']['rgb'], (160, 120)),
                             cv2.resize(cams['right_camera']['rgb'], (160, 120))], axis=1)
    image = resize_with_padding(np.concatenate([cams['head_camera']['rgb'], bottom], axis=0), size)
    return image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def save_frames_png(path, gt, pred, title):
    n = len(gt)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for j in range(n):
        for row, imgs, label in [(0, gt, 'GT'), (1, pred, 'Pred')]:
            axes[row, j].imshow(np.clip(imgs[j], 0, 1))
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            if j == 0:
                axes[row, j].set_ylabel(label, fontsize=12)
        axes[0, j].set_title('t (cond)' if j == 0 else f'+{j}', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_actions_png(path, state, gt, pred, title):
    fig, axes = plt.subplots(2, 7, figsize=(18, 5), sharex=True)
    steps = np.arange(len(gt) + 1)
    for k, ax in enumerate(axes.flat):
        ax.plot(steps, np.r_[state[k], gt[:, k]], 'k-', lw=2, label='GT')
        ax.plot(steps, np.r_[state[k], pred[:, k]], 'r--', lw=1.5, label='Motus')
        ax.set_title(JOINT_NAMES[k], fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--wan_path', required=True)
    parser.add_argument('--vlm_path', required=True)
    parser.add_argument('--task_dir', required=True, help='Contains aloha-agilex_clean_50/ and aloha-agilex_randomized_500/')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--subsets', nargs='+', default=['aloha-agilex_clean_50', 'aloha-agilex_randomized_500'])
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--positions', type=float, nargs='+', default=[0.0, 0.33, 0.66], help='Condition frame as fraction of the valid range')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_name = Path(args.task_dir).name

    policy = get_model({'ckpt_setting': args.ckpt, 'wan_path': args.wan_path, 'vlm_path': args.vlm_path,
                        'task_name': task_name, 'log_dir': str(out_dir / 'policy_logs')})
    common = policy.config_dict['common']
    ds, ratio, n_video = common['global_downsample_rate'], common['video_action_freq_ratio'], common['num_video_frames']
    chunk = n_video * ratio
    size = (common['video_height'], common['video_width'])

    # Capture predicted frames instead of only writing the policy's grid image
    captured = {}
    policy._save_frame_grid = lambda cond, pred: captured.update(pred=pred)

    results = []
    for subset in args.subsets:
        for ep in args.episodes:
            h5_path = Path(args.task_dir) / subset / 'data' / f'episode{ep}.hdf5'
            with open(Path(args.task_dir) / subset / 'instructions' / f'episode{ep}.json') as f:
                instruction = json.load(f)['seen'][0]
            with h5py.File(h5_path, 'r') as f:
                joints = f['joint_action/vector'][()].astype(np.float32)
            T = len(joints)
            max_t = T - chunk * ds - 1
            for pos in args.positions:
                t = int(round(pos * max(max_t, 0)))
                action_idx = [min(t + (i + 1) * ds, T - 1) for i in range(chunk)]
                video_idx = [action_idx[(j + 1) * ratio - 1] for j in range(n_video)]

                obs = load_obs(str(h5_path), t)
                reset_model(policy)
                policy.set_instruction(instruction)
                policy.update_obs(obs)
                captured.clear()
                torch.manual_seed(args.seed)
                policy.get_action()
                pred_actions = np.stack(list(policy.action_cache))[:chunk]
                pred_frames = captured['pred'].permute(0, 2, 3, 1).float().cpu().numpy() if 'pred' in captured else None

                state, gt_actions = joints[t], joints[action_idx]
                rec = {
                    'subset': subset, 'episode': ep, 't': t, 'episode_len': T, 'instruction': instruction,
                    'action_mae': float(np.abs(pred_actions - gt_actions).mean()),
                    'action_mae_hold_baseline': float(np.abs(state[None] - gt_actions).mean()),
                    'action_mae_per_joint': np.abs(pred_actions - gt_actions).mean(0).round(4).tolist(),
                }
                tag = f'{subset.split("_")[1]}_ep{ep}_t{t}'
                title = f'{task_name} | {subset} ep{ep} t={t}/{T} | "{instruction}"'
                save_actions_png(out_dir / f'actions_{tag}.png', state, gt_actions, pred_actions,
                                 f'{title}\nMAE Motus={rec["action_mae"]:.4f} vs hold-state={rec["action_mae_hold_baseline"]:.4f} rad')

                if pred_frames is not None:
                    cond = compose_view(obs, size)
                    gt_frames = [compose_view(load_obs(str(h5_path), i), size) for i in video_idx]
                    rec['video_psnr'] = float(np.mean([psnr(p, g) for p, g in zip(pred_frames, gt_frames)]))
                    rec['video_psnr_copy_baseline'] = float(np.mean([psnr(cond, g) for g in gt_frames]))
                    save_frames_png(out_dir / f'frames_{tag}.png', [cond] + gt_frames, [cond] + list(pred_frames),
                                    f'{title}\nPSNR Motus={rec["video_psnr"]:.2f} dB vs copy-frame={rec["video_psnr_copy_baseline"]:.2f} dB')
                    gif = [(np.concatenate([g, np.ones((size[0], 8, 3)), np.clip(p, 0, 1)], axis=1) * 255).astype(np.uint8)
                           for g, p in zip([cond] + gt_frames, [cond] + list(pred_frames))]
                    imageio.mimsave(out_dir / f'rollout_{tag}.gif', gif, duration=0.4, loop=0)

                results.append(rec)
                print(json.dumps({k: v for k, v in rec.items() if k != 'action_mae_per_joint'}), flush=True)

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    lines = [f'# Open-loop results: {task_name}', '',
             '| subset | ep | t | action MAE Motus | MAE hold-state | video PSNR Motus | PSNR copy-frame |',
             '|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['subset']} | {r['episode']} | {r['t']} | {r['action_mae']:.4f} | {r['action_mae_hold_baseline']:.4f} | "
                     f"{r.get('video_psnr', float('nan')):.2f} | {r.get('video_psnr_copy_baseline', float('nan')):.2f} |")
    for subset in args.subsets:
        rs = [r for r in results if r['subset'] == subset]
        lines.append(f"| **{subset} mean** | | | **{np.mean([r['action_mae'] for r in rs]):.4f}** | {np.mean([r['action_mae_hold_baseline'] for r in rs]):.4f} | "
                     f"**{np.mean([r.get('video_psnr', np.nan) for r in rs]):.2f}** | {np.mean([r.get('video_psnr_copy_baseline', np.nan) for r in rs]):.2f} |")
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
