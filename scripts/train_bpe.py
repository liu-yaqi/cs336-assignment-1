import argparse
import random
import time
from pathlib import Path
from time import perf_counter

from cs336_basics.tokenizer import BPETrainer, Tokenizer


def train_bpe(
    input_path: str | Path,
    vocab_size: int = 10000,
    special_tokens: list[str] | None = None,
    vocab_output_path: str | Path = "data/tinystories_vocab_10k.txt",
    merges_output_path: str | Path = "data/tinystories_merges_10k.txt",
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE on TinyStories and save vocab/merges in from_files format."""
    trainer = BPETrainer(vocab_size)
    vocab, merges = trainer.train(str(input_path), special_tokens or ["<|endoftext|>"])
    trainer.save(vocab_output_path, merges_output_path)
    return vocab, merges


def run_encode_decode_smoke_test(tokenizer: Tokenizer, text: str) -> None:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"[smoke] sample bytes={len(text.encode('utf-8'))}, tokens={len(encoded)}")
    print(f"[smoke] decoded matches input: {decoded == text}")


def sample_documents(input_path: str | Path, num_docs: int, seed: int) -> list[str]:
    """Reservoir-sample non-empty documents (one line = one document)."""
    rng = random.Random(seed)
    sampled: list[str] = []
    seen = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = line.strip()
            if not doc:
                continue
            seen += 1
            if len(sampled) < num_docs:
                sampled.append(doc)
            else:
                j = rng.randint(1, seen)
                if j <= num_docs:
                    sampled[j - 1] = doc

    if len(sampled) < num_docs:
        raise ValueError(f"Not enough non-empty docs in {input_path}. Need {num_docs}, got {len(sampled)}")
    return sampled


def evaluate_tokenizer_on_docs(tokenizer: Tokenizer, docs: list[str], repeats: int) -> tuple[float, float, int, int]:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))

    if total_tokens == 0:
        raise ValueError("Tokenizer produced zero tokens; cannot compute bytes/token.")

    compression_ratio = total_bytes / total_tokens

    payload = "\n".join(docs)
    payload_bytes = len(payload.encode("utf-8"))
    start = perf_counter()
    for _ in range(max(1, repeats)):
        tokenizer.encode(payload)
    elapsed = perf_counter() - start
    throughput = (payload_bytes * max(1, repeats)) / max(elapsed, 1e-12)

    return compression_ratio, throughput, total_bytes, total_tokens


def load_tokenizer(vocab_path: str | Path, merges_path: str | Path, special_tokens: list[str]) -> Tokenizer:
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    if not Path(merges_path).exists():
        raise FileNotFoundError(f"Missing merges file: {merges_path}")
    return Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)


def run_compression_report(
    tiny_tokenizer: Tokenizer,
    tiny_docs_path: str | Path,
    num_docs: int,
    seed: int,
    repeats: int,
) -> None:
    tiny_docs = sample_documents(tiny_docs_path, num_docs=num_docs, seed=seed)

    tiny_ratio, tiny_tp, tiny_bytes, tiny_tokens = evaluate_tokenizer_on_docs(tiny_tokenizer, tiny_docs, repeats)

    print("[report] TinyStories tokenizer (10K) on TinyStories sample")
    print(f"[report] docs={num_docs} bytes={tiny_bytes} tokens={tiny_tokens}")
    print(f"[report] compression(bytes/token)={tiny_ratio:.6f}")
    print(f"[report] throughput(bytes/s)={tiny_tp:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 10k BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to TinyStories training corpus.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--vocab-output",
        type=str,
        default="data/tinystories_vocab_10k.txt",
        help="Output path for vocab file.",
    )
    parser.add_argument(
        "--merges-output",
        type=str,
        default="data/tinystories_merges_10k.txt",
        help="Output path for merges file.",
    )
    parser.add_argument("--benchmark-only", action="store_true", help="Skip training and only run report.")
    parser.add_argument("--run-compression-report", action="store_true", help="Compute bytes/token and bytes/s.")
    parser.add_argument("--num-docs", type=int, default=10, help="Number of sampled documents per corpus.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for document sampling.")
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=20,
        help="Repeat count for throughput estimation.",
    )
    parser.add_argument(
        "--tiny-docs-path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="TinyStories corpus path for report sampling.",
    )
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    tiny_tokenizer: Tokenizer

    if args.benchmark_only:
        tiny_tokenizer = load_tokenizer(args.vocab_output, args.merges_output, special_tokens)
    else:
        start = time.time()
        vocab, merges = train_bpe(
            input_path=args.input_path,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            vocab_output_path=args.vocab_output,
            merges_output_path=args.merges_output,
        )
        print(f"[train] elapsed_s={time.time() - start:.2f}")
        print(f"[train] vocab size: {len(vocab)}, merges: {len(merges)}")
        print(f"[train] saved vocab to: {args.vocab_output}")
        print(f"[train] saved merges to: {args.merges_output}")

        tiny_tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    smoke_text = "Once upon a time, there was a tiny story. <|endoftext|>"
    run_encode_decode_smoke_test(tiny_tokenizer, smoke_text)


    if args.run_compression_report:
        run_compression_report(
            tiny_tokenizer=tiny_tokenizer,
            tiny_docs_path=args.tiny_docs_path,
            num_docs=args.num_docs,
            seed=args.sample_seed,
            repeats=args.benchmark_repeats,
        )


if __name__ == "__main__":
    main()
