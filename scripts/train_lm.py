import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.basic_module import CrossEntropy, TransformerLM, generate_tokens
from cs336_basics.data import Dataset
from cs336_basics.optimizer import SimpleAdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransformerLM with memmap token data.")

    # Data
    parser.add_argument("--train-data-path", type=str, required=True, help="Path to train token binary file.")
    parser.add_argument("--valid-data-path", type=str, required=True, help="Path to valid token binary file.")
    parser.add_argument(
        "--data-dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32", "int64"],
        help="dtype used by token binary files.",
    )

    # Model
    parser.add_argument("--vocab-size", type=int, default=50527)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--theta-base", type=float, default=10000.0)
    parser.add_argument("--rms-eps", type=float, default=1e-8)

    # Optimizer and schedule
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--cosine-cycle-steps", type=int, default=20000) # total
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Loop
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)

    # Runtime and output
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cs336-basics")

    # Generation from best checkpoint
    parser.add_argument("--tokenizer-vocab-path", type=str, default="data/fromvalid/tinystories_vocab_10k.txt")
    parser.add_argument("--tokenizer-merges-path", type=str, default="data/fromvalid/tinystories_merges_10k.txt")
    parser.add_argument("--gen-prompt", type=str, default="last year, i went to a ")
    parser.add_argument("--gen-max-new-tokens", type=int, default=80)
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--gen-top-p", type=float, default=0.9)

    return parser.parse_args()


def _dtype_from_string(dtype_name: str) -> np.dtype:
    dtype_map = {
        "uint16": np.uint16,
        "int32": np.int32,
        "int64": np.int64,
    }
    return dtype_map[dtype_name]


def load_memmap_dataset(path: str | os.PathLike, dtype: np.dtype) -> Dataset:
    token_array = np.memmap(path, dtype=dtype, mode="r")
    return Dataset(token_array)


def _global_parameter_norm(model: torch.nn.Module) -> float:
    total_sq = 0.0
    for param in model.parameters():
        total_sq += float(param.detach().pow(2).sum().item())
    return float(total_sq**0.5)


def _global_gradient_norm(model: torch.nn.Module) -> float:
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total_sq += float(param.grad.detach().pow(2).sum().item())
    return float(total_sq**0.5)


def _loss_to_perplexity(loss_value: float) -> float:
    # Clamp exponent input to keep perplexity finite for unstable runs.
    return float(math.exp(min(loss_value, 20.0)))


def evaluate(
    model: TransformerLM,
    loss_fn: CrossEntropy,
    dataset: Dataset,
    batch_size: int,
    context_length: int,
    device: str,
    eval_steps: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = dataset.get_batch(batch_size=batch_size, context_length=context_length, device=device)
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def save_checkpoint(
    checkpoint_path: Path,
    model: TransformerLM,
    optimizer: SimpleAdamW,
    step: int,
    train_loss: float,
    valid_loss: float,
    config: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "config": config,
        },
        checkpoint_path,
    )


def generate_story_from_best(args: argparse.Namespace, run_dir: Path) -> None:
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        print(f"[generation] skipped: best checkpoint not found at {best_path}")
        return

    vocab_path = Path(args.tokenizer_vocab_path)
    merges_path = Path(args.tokenizer_merges_path)
    if not vocab_path.exists() or not merges_path.exists():
        print(
            "[generation] skipped: tokenizer files missing. "
            f"vocab={vocab_path} merges={merges_path}"
        )
        return

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
    )

    checkpoint = torch.load(best_path, map_location=args.device)
    best_model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        theta_base=args.theta_base,
        eps=args.rms_eps,
    ).to(args.device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    prompt_token_ids = tokenizer.encode(args.gen_prompt)
    if len(prompt_token_ids) == 0:
        print("[generation] skipped: encoded prompt is empty")
        return

    end_token_id = tokenizer.dstoi.get(b"<|endoftext|>")
    generated_ids = generate_tokens(
        model=best_model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        end_token_id=end_token_id,
        device=args.device,
    )
    story = tokenizer.decode(generated_ids)

    print(f"[generation] prompt: {args.gen_prompt}")
    print("[generation] story:")
    print(story)

    output_path = run_dir / "best_model_story.txt"
    output_path.write_text(story, encoding="utf-8")
    print(f"[generation] saved to {output_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = _dtype_from_string(args.data_dtype)
    train_dataset = load_memmap_dataset(args.train_data_path, dtype)
    valid_dataset = load_memmap_dataset(args.valid_data_path, dtype)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        theta_base=args.theta_base,
        eps=args.rms_eps,
    ).to(args.device)

    optimizer = SimpleAdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    loss_fn = CrossEntropy()

    checkpoint_dir = Path(args.checkpoint_dir)
    run_dir = checkpoint_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, 
                                   name=args.run_name, 
                                   config=config)
        except Exception as exc:
            print(f"[warn] wandb disabled: {exc}")

    best_valid = math.inf
    for step in range(1, args.train_steps + 1):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
            cosine_cycle_steps=args.cosine_cycle_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = train_dataset.get_batch(
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device,
        )
        logits = model(x)
        loss = loss_fn(logits, y)
        model_output_norm = float(logits.detach().norm().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm_before_clip = _global_gradient_norm(model)
        gradient_clipping(model.parameters(), args.max_grad_norm)
        grad_norm_after_clip = _global_gradient_norm(model)

        optimizer.step()
        param_norm = _global_parameter_norm(model)

        train_loss = float(loss.item())

        if step % args.log_interval == 0:
            print(
                "step="
                f"{step} "
                f"train_loss={train_loss:.4f} "
                f"lr={lr:.6e} "
                f"model_output_norm={model_output_norm:.4f} "
                f"grad_norm_pre={grad_norm_before_clip:.4f} "
                f"grad_norm_post={grad_norm_after_clip:.4f} "
                f"param_norm={param_norm:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "train_loss": train_loss,
                        "lr": lr,
                        "model_output_norm": model_output_norm,
                        "grad_norm_pre_clip": grad_norm_before_clip,
                        "grad_norm_post_clip": grad_norm_after_clip,
                        "param_norm": param_norm,
                    }
                )

        if step % args.eval_interval == 0 or step == args.train_steps:
            valid_loss = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataset=valid_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_steps=args.eval_steps,
            )
            valid_perplexity = _loss_to_perplexity(valid_loss)
            print(f"[eval] step={step} valid_loss={valid_loss:.4f} valid_ppl={valid_perplexity:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "valid_loss": valid_loss,
                        "valid_perplexity": valid_perplexity,
                    }
                )

            if valid_loss < best_valid:
                best_valid = valid_loss
                save_checkpoint(
                    checkpoint_path=run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=train_loss,
                    valid_loss=valid_loss,
                    config=config,
                )

    generate_story_from_best(args=args, run_dir=run_dir)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
