"""
Pre-encode RoboTwin instructions with the WAN T5 encoder on GPU (run before loading Motus),
so evaluation can use MOTUS_T5_CACHE=<output> and never runs T5 on CPU.

Usage:
    python encode_t5_cache.py --wan_path pretrained_models/Wan2.2-TI2V-5B --output t5_cache.pt \
        --instructions "/path/to/task/*/instructions/*.json" [--keys seen unseen]
"""

import argparse
import glob
import json
import os

import torch

from deploy_policy import SCENE_PREFIX, MmapT5EncoderModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wan_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--instructions', nargs='+', required=True, help='Instruction JSON files/globs ({"seen": [...], "unseen": [...]}) or .txt (one per line)')
    parser.add_argument('--keys', nargs='+', default=['seen', 'unseen'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    texts = set()
    for pattern in args.instructions:
        for path in glob.glob(pattern):
            if path.endswith('.json'):
                with open(path) as f:
                    data = json.load(f)
                for key in args.keys:
                    texts.update(data.get(key, []))
            else:
                with open(path) as f:
                    texts.update(line.strip() for line in f if line.strip())

    cache = torch.load(args.output) if os.path.exists(args.output) else {}
    todo = sorted(f"{SCENE_PREFIX}{t}" for t in texts if f"{SCENE_PREFIX}{t}" not in cache)
    print(f"{len(texts)} instructions, {len(cache)} already cached, encoding {len(todo)}")

    encoder = MmapT5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device='cuda',
        checkpoint_path=os.path.join(args.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=os.path.join(args.wan_path, 'google', 'umt5-xxl'),
    )
    with torch.no_grad():
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            for text, emb in zip(batch, encoder(batch, 'cuda')):
                cache[text] = [emb.cpu()]

    torch.save(cache, args.output)
    print(f"Saved {len(cache)} embeddings to {args.output}")


if __name__ == '__main__':
    main()
