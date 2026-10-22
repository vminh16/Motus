"""
Standalone smoke test for the RoboTwin Motus policy (no simulator needed).
Loads the policy, feeds one observation (from a RoboTwin HDF5 episode if given, otherwise random),
and reports action shape, latency, peak VRAM and host RAM.

Usage:
    python smoke_test.py --ckpt pretrained_models/Motus_robotwin2 --wan_path pretrained_models/Wan2.2-TI2V-5B \
        --vlm_path pretrained_models/Qwen3-VL-2B-Instruct [--hdf5 episode0.hdf5] [--steps 3]
"""

import argparse
import resource
import time

import cv2
import numpy as np
import torch

from deploy_policy import get_model, reset_model


def load_obs(hdf5_path, frame_idx=0):
    if hdf5_path is None:
        rng = np.random.default_rng(0)
        cam = lambda: {'rgb': rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)}
        return {'observation': {'head_camera': cam(), 'left_camera': cam(), 'right_camera': cam()},
                'joint_action': {'vector': np.zeros(14, dtype=np.float32)}}
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        cams = {}
        for name in ['head_camera', 'left_camera', 'right_camera']:
            buf = f[f'observation/{name}/rgb'][frame_idx]
            cams[name] = {'rgb': cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)}
        state = f['joint_action/vector'][frame_idx].astype(np.float32)
    return {'observation': cams, 'joint_action': {'vector': state}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--wan_path', required=True)
    parser.add_argument('--vlm_path', required=True)
    parser.add_argument('--hdf5', default=None)
    parser.add_argument('--instruction', default='Click the alarm clock.')
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    policy = get_model({'ckpt_setting': args.ckpt, 'wan_path': args.wan_path, 'vlm_path': args.vlm_path,
                        'task_name': 'smoke_test'})
    print(f'[load] {time.time() - t0:.1f}s, VRAM allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB')

    reset_model(policy)
    policy.set_instruction(args.instruction)
    obs = load_obs(args.hdf5)
    for step in range(args.steps):
        policy.update_obs(obs)
        policy.action_cache.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        policy.get_action()
        torch.cuda.synchronize()
        actions = np.stack(list(policy.action_cache))
        print(f'[step {step}] {time.time() - t0:.2f}s, actions {actions.shape}, '
              f'range [{actions.min():.3f}, {actions.max():.3f}], finite={np.isfinite(actions).all()}')

    print(f'[peak] VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, '
          f'host RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB')


if __name__ == '__main__':
    main()
