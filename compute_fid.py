# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compute FID directly with TorchMetrics, using real images from an ImageFolder
and fake images sampled from a SiT checkpoint.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from diffusers.models import AutoencoderKL

from download import find_model
from models import SiT_models
from sample_ddp import infer_learn_sigma_from_state_dict
from train import center_crop_arr
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import Sampler, create_transport


def import_fid_metric():
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing FID dependencies. Install them in venv_sit with:\n"
            "  /workspace/venv_sit/bin/pip install 'torchmetrics[image]' torch-fidelity\n"
        ) from exc
    return FrechetInceptionDistance


def setup_dist():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_dist(enabled):
    if enabled:
        dist.destroy_process_group()


def local_target(total, rank, world_size):
    base = total // world_size
    remainder = total % world_size
    return base + (1 if rank < remainder else 0)


def build_real_loader(args, rank, world_size):
    transform_list = []
    if args.center_crop:
        transform_list.append(transforms.Lambda(lambda image: center_crop_arr(image, args.image_size)))
    elif args.resize_real:
        transform_list.append(transforms.Resize((args.image_size, args.image_size)))
    transform_list.extend([transforms.Lambda(lambda image: image.convert("RGB")), transforms.ToTensor()])

    dataset = ImageFolder(args.data_path, transform=transforms.Compose(transform_list))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_model(args, device):
    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size == 256, "Auto-download is only wired for ImageNet 256."
        state_dict = find_model(f"SiT-XL-2-{args.image_size}x{args.image_size}.pt")
        learn_sigma = True
    else:
        state_dict = find_model(args.ckpt)
        learn_sigma = infer_learn_sigma_from_state_dict(state_dict)

    model = SiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        use_null_label=args.use_null_label,
        learn_sigma=learn_sigma,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_sample_fn(mode, args):
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        return sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse,
        )
    return sampler.sample_sde(
        sampling_method=args.sampling_method,
        diffusion_form=args.diffusion_form,
        diffusion_norm=args.diffusion_norm,
        last_step=args.last_step,
        last_step_size=args.last_step_size,
        num_steps=args.num_sampling_steps,
    )


@torch.inference_mode()
def sample_fake_uint8(model, vae, sample_fn, args, device, batch_size):
    latent_size = args.image_size // 8
    z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
    using_cfg = args.cfg_scale > 1.0 and not args.use_null_label

    if args.use_null_label:
        model_kwargs = {}
        model_fn = model.forward
    elif using_cfg:
        y = torch.randint(0, args.num_classes, (batch_size,), device=device)
        z = torch.cat([z, z], dim=0)
        y_null = torch.full((batch_size,), args.num_classes, device=device)
        model_kwargs = {"y": torch.cat([y, y_null], dim=0), "cfg_scale": args.cfg_scale}
        model_fn = model.forward_with_cfg
    else:
        y = torch.randint(0, args.num_classes, (batch_size,), device=device)
        model_kwargs = {"y": y}
        model_fn = model.forward

    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)

    samples = vae.decode(samples / 0.18215).sample
    return torch.clamp(127.5 * samples + 128.0, 0, 255).to(torch.uint8)


def to_fid_float01(x):
    if x.dtype == torch.uint8:
        x = x.float().div(255.0)
    return x.clamp(0.0, 1.0)


def to_uint8_nchw(x):
    if x.dtype == torch.uint8:
        return x.detach().cpu()
    return (x.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def save_grid_png_uint8(images_uint8_nchw, out_path):
    from PIL import Image
    from torchvision.utils import make_grid

    if images_uint8_nchw.ndim != 4:
        raise ValueError(f"Expected images of shape (N,C,H,W), got {tuple(images_uint8_nchw.shape)}")
    if images_uint8_nchw.shape[0] == 0:
        raise ValueError("Cannot save empty snapshot grid.")

    images = images_uint8_nchw.detach().cpu().float().div(255.0)
    nrow = max(1, int(math.ceil(math.sqrt(images.shape[0]))))
    grid = make_grid(images, nrow=nrow, padding=2)
    grid = (grid.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(out_path)


def snapshots_paths(args, mode):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = Path(args.ckpt) if args.ckpt is not None else Path("pretrained")
    root_dir = Path(args.report_dir) if args.report_dir is not None else ckpt.parent
    root = root_dir / f"snapshots_{timestamp}_{mode.lower()}_steps{args.num_sampling_steps}_{args.snapshots}samples"
    root.mkdir(parents=True, exist_ok=True)
    return root, root / "real.png", root / "fake.png"


def report_path(args, mode, num_samples, snapshots_root=None):
    if snapshots_root is not None:
        return snapshots_root / "fid_result.yaml"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = Path(args.ckpt) if args.ckpt is not None else Path("pretrained")
    report_dir = Path(args.report_dir) if args.report_dir is not None else ckpt.parent
    return report_dir / f"fid_{timestamp}_{mode.lower()}_steps{args.num_sampling_steps}_{num_samples}samples.yaml"


def write_report(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = make_json_safe(payload)
    try:
        import yaml

        with path.open("w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    except ModuleNotFoundError:
        with path.open("w") as f:
            json.dump(payload, f, indent=2)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if type(value) in {str, int, float, bool} or value is None:
        return value
    return str(value)


def sampler_report(mode, args):
    report = {
        "mode": mode,
        "path_type": args.path_type,
        "prediction": args.prediction,
        "loss_weight": args.loss_weight,
        "train_eps": args.train_eps,
        "sample_eps": args.sample_eps,
        "num_sampling_steps": int(args.num_sampling_steps),
    }
    if mode == "ODE":
        report.update(
            {
                "sampling_method": args.sampling_method,
                "rtol": float(args.rtol),
                "atol": float(args.atol),
                "reverse": bool(args.reverse),
                "adaptive_internal_nfe_unreported": args.sampling_method.lower() in {"dopri5", "dopri8", "adaptive_heun"},
            }
        )
    else:
        report.update(
            {
                "sampling_method": args.sampling_method,
                "diffusion_form": args.diffusion_form,
                "diffusion_norm": float(args.diffusion_norm),
                "last_step": args.last_step,
                "last_step_size": float(args.last_step_size),
            }
        )
    return report


def build_report(
    args,
    mode,
    fid_value,
    num_samples_processed,
    use_dist,
    world_size,
    device,
    snapshots_root=None,
    snapshots_saved=0,
):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fid_value": float(fid_value),
        "fid_config": {
            "checkpoint": args.ckpt,
            "data_path": args.data_path,
            "num_samples_requested": int(args.num_fid_samples),
            "num_samples_processed": int(num_samples_processed),
            "feature": int(args.feature),
            "batch_size_per_process": int(args.per_proc_batch_size),
            "num_workers": int(args.num_workers),
            "seed": int(args.global_seed),
            "model": {
                "name": args.model,
                "image_size": int(args.image_size),
                "num_classes": int(args.num_classes),
                "use_null_label": bool(args.use_null_label),
                "cfg_scale": float(args.cfg_scale),
                "vae": args.vae,
            },
            "real_preprocessing": {
                "center_crop": bool(args.center_crop),
                "resize_real": bool(args.resize_real),
                "to_rgb": True,
                "to_float01": True,
            },
            "sampler": sampler_report(mode, args),
            "torch": {
                "version": str(torch.__version__),
                "tf32": bool(args.tf32),
            },
            "distributed": {
                "enabled": bool(use_dist),
                "world_size": int(world_size),
            },
            "snapshots": {
                "target": int(args.snapshots),
                "saved": int(snapshots_saved),
                "root": str(snapshots_root) if snapshots_root is not None else None,
                "real": str(snapshots_root / "real.png") if snapshots_root is not None else None,
                "fake": str(snapshots_root / "fake.png") if snapshots_root is not None else None,
            },
            "resolved_device": str(device),
            "command": " ".join(sys.argv),
        },
    }


def main(mode, args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "compute_fid.py currently requires CUDA."

    use_dist, rank, world_size, local_rank = setup_dist()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.global_seed * world_size + rank)

    FrechetInceptionDistance = import_fid_metric()
    model = build_model(args, device)
    sample_fn = build_sample_fn(mode, args)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    loader = build_real_loader(args, rank, world_size)
    fid = FrechetInceptionDistance(feature=args.feature, normalize=True).to(device)
    fid.set_dtype(torch.float64)

    target = local_target(args.num_fid_samples, rank, world_size)
    seen = 0
    total_iters = math.ceil(target / args.per_proc_batch_size)
    pbar = tqdm(total=args.num_fid_samples, desc="FID", unit="img") if is_main else None
    snapshots_root = snapshots_real_file = snapshots_fake_file = None
    snapshots_real = []
    snapshots_fake = []
    snapshots_saved = 0
    if is_main and args.snapshots > 0:
        snapshots_root, snapshots_real_file, snapshots_fake_file = snapshots_paths(args, mode)
        print(f"Saving {args.snapshots} snapshots as grids: {snapshots_root}/real.png and fake.png")

    for real, _ in loader:
        if seen >= target:
            break
        take = min(real.shape[0], target - seen)
        real = real[:take].to(device, non_blocking=True)
        fake = sample_fake_uint8(model, vae, sample_fn, args, device, take)

        if is_main and args.snapshots > 0 and snapshots_saved < args.snapshots:
            to_save = min(args.snapshots - snapshots_saved, take)
            snapshots_real.append(to_uint8_nchw(real[:to_save]))
            snapshots_fake.append(to_uint8_nchw(fake[:to_save]))
            snapshots_saved += to_save

        fid.update(to_fid_float01(real), real=True)
        fid.update(to_fid_float01(fake), real=False)
        seen += take

        if use_dist:
            seen_tensor = torch.tensor([seen], device=device, dtype=torch.long)
            dist.all_reduce(seen_tensor, op=dist.ReduceOp.SUM)
            global_seen = int(seen_tensor.item())
        else:
            global_seen = seen
        if pbar is not None and global_seen > pbar.n:
            pbar.update(global_seen - pbar.n)

        if seen >= target:
            break

    if pbar is not None:
        pbar.close()

    seen_tensor = torch.tensor([seen], device=device, dtype=torch.long)
    if use_dist:
        dist.all_reduce(seen_tensor, op=dist.ReduceOp.SUM)
    global_seen = int(seen_tensor.item())
    if global_seen == 0:
        raise RuntimeError("No images were processed. Check --data-path.")

    fid_value = float(fid.compute().detach().cpu())
    if is_main:
        if args.snapshots > 0 and snapshots_saved > 0:
            assert snapshots_root is not None
            assert snapshots_real_file is not None
            assert snapshots_fake_file is not None
            save_grid_png_uint8(torch.cat(snapshots_real, dim=0)[:snapshots_saved], snapshots_real_file)
            save_grid_png_uint8(torch.cat(snapshots_fake, dim=0)[:snapshots_saved], snapshots_fake_file)
            print(f"Snapshots saved as grid images: {snapshots_saved}/{args.snapshots}")
        print(f"FID ({mode}, n={global_seen}, feature={args.feature}): {fid_value:.6f}")
        print(
            "Sampler: "
            f"{args.sampling_method}, steps={args.num_sampling_steps}, "
            f"rtol={getattr(args, 'rtol', None)}, atol={getattr(args, 'atol', None)}"
        )
        if not args.no_report:
            report = build_report(
                args,
                mode,
                fid_value,
                global_seen,
                use_dist,
                world_size,
                device,
                snapshots_root=snapshots_root,
                snapshots_saved=snapshots_saved,
            )
            path = report_path(args, mode, global_seen, snapshots_root=snapshots_root)
            write_report(path, report)
            print(f"Saved FID report to: {path}")

    cleanup_dist(use_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if len(sys.argv) < 2:
        print("Usage: compute_fid.py <ODE|SDE> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--use-null-label", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature", type=int, default=2048)
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--resize-real", action="store_true")
    parser.add_argument("--snapshots", type=int, default=0)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None)

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    else:
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    main(mode, args)
