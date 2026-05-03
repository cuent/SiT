from pathlib import Path
import json
import pandas as pd
import shlex

WANDB_DIR = Path("wandb")

def get_arg_value(args, key):
    for i, arg in enumerate(args):
        arg = str(arg)

        if arg == key and i + 1 < len(args):
            return args[i + 1]

        if arg.startswith(key + "="):
            return arg.split("=", 1)[1]

    return None


records = []

for metadata_path in WANDB_DIR.glob("run-*/files/wandb-metadata.json"):
    with metadata_path.open("r") as f:
        meta = json.load(f)

    args = meta.get("args", [])
    program = meta.get("program") or meta.get("codePath") or ""

    command = " ".join([shlex.quote(program)] + [shlex.quote(str(a)) for a in args])

    gpu_nvidia = meta.get("gpu_nvidia", [])
    gpu_names = sorted({gpu.get("name", "") for gpu in gpu_nvidia if gpu.get("name")})

    records.append({
        "run_folder": metadata_path.parents[1].name,
        "metadata_path": str(metadata_path),
        "startedAt": meta.get("startedAt"),
        "command": command,
        "gpu_count": meta.get("gpu_count"),
        "gpu_names": ", ".join(gpu_names),
        "global_batch_size": get_arg_value(args, "--global-batch-size"),
        "data_path": get_arg_value(args, "--data-path"),
        "ckpt": get_arg_value(args, "--ckpt"),
        "start_step": get_arg_value(args, "--start-step"),
        "wandb_run_id": get_arg_value(args, "--wandb-run-id"),
        "cfg_scale": get_arg_value(args, "--cfg-scale"),
        "num_classes": get_arg_value(args, "--num-classes"),
        "cudaVersion": meta.get("cudaVersion"),
        "host": meta.get("host"),
        "git_commit": meta.get("git", {}).get("commit"),
    })

df_meta = pd.DataFrame(records)

df_meta = df_meta.sort_values("startedAt").reset_index(drop=True)

df_meta[
    [
        "run_folder",
        "startedAt",
        "gpu_count",
        "global_batch_size",
        "start_step",
        "wandb_run_id",
        "command",
    ]
]


for row in df_meta["command"]:
    print(row)