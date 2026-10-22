"""
Check the CPU T5 path: latency per new instruction, and that trimming padding matches the GPU full-padding embeddings
stored by encode_t5_cache.py.

Usage:
    python bench_t5_cpu.py --wan_path pretrained_models/Wan2.2-TI2V-5B --cache t5_cache.pt [--n 3]
"""

import argparse
import os
import time

import torch

from deploy_policy import MmapT5EncoderModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wan_path', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    cache = torch.load(args.cache)
    encoder = MmapT5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device='cpu',
        checkpoint_path=os.path.join(args.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=os.path.join(args.wan_path, 'google', 'umt5-xxl'),
    )
    with torch.no_grad():
        for text in list(cache)[:args.n]:
            t0 = time.time()
            emb = encoder([text], 'cpu')[0].float()
            ref = cache[text][0].float()
            cos = torch.nn.functional.cosine_similarity(emb, ref, dim=-1)
            print(f'{time.time() - t0:.1f}s  tokens={emb.shape[0]}  shape_match={emb.shape == ref.shape}  '
                  f'cos min={cos.min():.5f} mean={cos.mean():.5f}  max_abs_diff={(emb - ref).abs().max():.4f}')


if __name__ == '__main__':
    main()
