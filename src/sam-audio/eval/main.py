# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import argparse
import json
import os

import pandas as pd
import torch
import torch.distributed as dist
from dataset import SETTINGS, make_dataset
from metrics import CLAP, Aesthetic, ImageBind, Judge
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from sam_audio import SAMAudio, SAMAudioProcessor


def gather_and_average_results(results, world_size):
    if world_size == 1:
        return json.loads(results.mean().to_json())

    # 1. Gather all dictionaries to all ranks
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(
        all_results, {"sum": results.sum().to_json(), "count": len(results)}
    )

    summed = {}
    counts = 0

    for res in all_results:
        for k, v in json.loads(res["sum"]).items():
            if k not in summed:
                summed[k] = 0.0
            summed[k] += v
        counts += res["count"]

    # 3. Compute average for keys that appeared at least once
    averaged = {k: summed[k] / counts for k in summed}

    return averaged


def main(
    settings: list[str],
    cache_path: str,
    batch_size: int,
    checkpoint_path: str,
    num_workers: int = 4,
    reranking_candidates: int = 8,
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    model = SAMAudio.from_pretrained(checkpoint_path)
    model = model.eval().to(device)
    processor = SAMAudioProcessor.from_pretrained(checkpoint_path)

    judge_metric = Judge(device=device)
    aes_metric = Aesthetic(device=device)
    clap_metric = CLAP(device=device)
    imagebind_metric = ImageBind(device=device)

    for setting in settings:
        print(f"Evaluating: {setting}")
        dset = make_dataset(setting, cache_path=cache_path, collate_fn=processor)
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(dset)

        dl = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dset.collate,
            num_workers=num_workers,
            sampler=sampler,
        )

        all_metrics = [
            judge_metric,
            aes_metric,
            clap_metric,
        ]

        if dset.visual:
            all_metrics.append(imagebind_metric)

        dfs = []
        with torch.inference_mode():
            for batch in tqdm(dl, disable=rank > 1):
                batch = batch.to(device)
                result = model.separate(
                    batch, reranking_candidates=reranking_candidates
                )
                mets = {}
                for metric in all_metrics:
                    input_wavs = model.unbatch(batch.audios.squeeze(1), batch.wav_sizes)

                    mets.update(
                        metric(
                            target_wavs=result.target,
                            target_wavs_sample_rate=model.sample_rate,
                            descriptions=batch.descriptions,
                            input_wavs=input_wavs,
                            videos=batch.masked_video,
                        )
                    )

                dfs.append(pd.DataFrame.from_dict(mets))

        df = pd.concat(dfs)
        averaged_results = gather_and_average_results(df, world_size)
        if rank == 0:
            results_dict = {k: f"{v:.3f}" for k, v in averaged_results.items()}
            print(json.dumps(results_dict, indent=4))
            os.makedirs("results", exist_ok=True)
            outfile = f"results/{setting}.json"
            with open(outfile, "w") as fout:
                print(json.dumps(results_dict), file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        "-s",
        choices=SETTINGS.keys(),
        help=f"Which setting to evaluate.  Choices: {SETTINGS.keys()}",
        default=["instr-pro"],
        nargs="+",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=os.path.expanduser("~/.cache/sam_audio"),
        help="Where to cache downloaded datasets",
    )
    parser.add_argument(
        "--checkpoint-path", "-p", type=str, default="facebook/sam-audio-large"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-workers", "-w", type=int, default=4, help="Number of workers"
    )
    parser.add_argument("--candidates", "-c", type=int, default=8)
    opt = parser.parse_args()
    main(
        settings=opt.setting,
        cache_path=opt.cache_path,
        batch_size=opt.batch_size,
        checkpoint_path=opt.checkpoint_path,
        num_workers=opt.num_workers,
        reranking_candidates=opt.candidates,
    )
