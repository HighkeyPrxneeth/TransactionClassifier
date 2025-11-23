#!/usr/bin/env python3
"""
Print a detailed Markdown summary of the model defined in `model.py`.

Usage:
  python scripts/model_summary.py [--checkpoint PATH] [--output FILE] [--top N]

The script prints a markdown report with per-module parameter counts, shapes,
total/ trainable parameters, and estimated memory footprint.
"""
import argparse
import sys
from pathlib import Path
import torch

# Import the project's model and config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import Config
from model import ModernTrajectoryNet


def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}EB"


def build_report(model: torch.nn.Module, cfg: Config, checkpoint_path: Path = None, top_k: int = 30):
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state, strict=False)
            loaded = True
        except Exception as e:
            loaded = False
            load_error = str(e)
    else:
        loaded = False

    # Totals
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    bytes_float32 = total_params * 4

    # Per-module breakdown (only direct parameters to avoid double counting)
    modules = []
    for name, module in model.named_modules():
        params = list(module.named_parameters(recurse=False))
        if not params:
            continue
        module_count = sum(p.numel() for _, p in params)
        param_entries = []
        for pname, p in params:
            full = f"{name + '.' if name else ''}{pname}"
            param_entries.append((full, tuple(p.shape), p.numel(), p.requires_grad))
        modules.append((name or '<root>', module_count, param_entries))

    # Sort modules by descending param count
    modules_sorted = sorted(modules, key=lambda x: x[1], reverse=True)

    # Begin markdown
    lines = []
    lines.append(f"# Model Summary: {model.__class__.__name__}\n")
    lines.append("## Configuration\n")
    try:
        cfg_items = vars(cfg)
    except Exception:
        cfg_items = { 'd_model': getattr(model, 'd_model', 'unknown'), 'n_layers': getattr(model, 'n_layers', 'unknown') }

    for k, v in cfg_items.items():
        lines.append(f"- **{k}**: `{v}`")

    lines.append("\n## Size & Parameters\n")
    lines.append(f"- **Total parameters:** `{total_params:,}`")
    lines.append(f"- **Trainable parameters:** `{trainable_params:,}`")
    lines.append(f"- **Non-trainable parameters:** `{non_trainable:,}`")
    lines.append(f"- **Estimated size (float32):** `{sizeof_fmt(bytes_float32)}`")
    if checkpoint_path is not None:
        lines.append(f"- **Checkpoint loaded:** `{str(checkpoint_path)}` -> `{loaded}`")
        if not loaded and 'load_error' in locals():
            lines.append(f"  - load error: `{load_error}`")

    lines.append("\n## Per-Module Parameter Breakdown\n")
    lines.append("| Module | # params | Example params (name: shape, trainable) |")
    lines.append("|---|---:|---|")

    for name, count, params in modules_sorted[:top_k]:
        details = []
        for n, shape, numel, req in params[:4]:
            details.append(f"`{n}`: `{shape}`{' ✓' if req else ' ✗' }")
        if len(params) > 4:
            details.append("...")
        lines.append(f"| `{name}` | `{count:,}` | {'<br>'.join(details)} |")

    if len(modules_sorted) > top_k:
        lines.append(f"\n_... {len(modules_sorted)-top_k} more modules omitted (use --top to increase)_\n")

    # Full parameter list (flat)
    lines.append("## Full Parameter List\n")
    lines.append("| Parameter | Shape | # params | Trainable |")
    lines.append("|---|---:|---:|---:|")
    for name, p in model.named_parameters():
        lines.append(f"| `{name}` | `{tuple(p.shape)}` | `{p.numel():,}` | {'Yes' if p.requires_grad else 'No'} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=Path, help='Path to checkpoint to load (optional)')
    parser.add_argument('--output', '-o', type=Path, help='Write markdown output to file (optional)')
    parser.add_argument('--top', type=int, default=30, help='Top-N modules to show in breakdown')
    parser.add_argument('--device', type=str, default='cpu', help='Device to move the model to for inspection')

    args = parser.parse_args()

    cfg = Config()
    model = ModernTrajectoryNet(cfg)
    model.to(args.device)

    md = build_report(model, cfg, checkpoint_path=args.checkpoint, top_k=args.top)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding='utf-8')
        print(f'Wrote model summary to {args.output}')
    else:
        print(md)


if __name__ == '__main__':
    main()
