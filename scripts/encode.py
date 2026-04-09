
import codecs
import os
from pathlib import Path

import numpy as np
import regex
# import tiktoken
from cs336_basics.tokenizer import Tokenizer


SPLIT_PAT = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


CONFIG = {
    "input_path": "data/TinyStoriesV2-GPT4-valid.txt",
    "output_path": "data/new/valid.bin",
    "dtype": "uint32",
    "chunk_size_mb": 8,
    "pretoken_batch_size": 8192,
    "flush_tokens": 2_000_000,
    "log_interval_chunks": 8,
}


def _dtype_from_name(dtype_name: str) -> np.dtype:
    dtype_map = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }
    return dtype_map[dtype_name]


def _extract_complete_pretokens(text: str, final: bool) -> tuple[list[str], str]:
    pretokens: list[str] = []
    last_end = 0
    for match in SPLIT_PAT.finditer(text, partial=not final):
        if match.partial:
            return pretokens, text[match.start():]

        token = match.group(0)
        if token:
            pretokens.append(token)
        last_end = match.end()

    return pretokens, text[last_end:]


def iter_pretokens_from_file(file_path: str, chunk_size_bytes: int):
    decoder = codecs.getincrementaldecoder("utf-8")()
    carry = ""

    with open(file_path, "rb") as f:
        while True:
            raw_chunk = f.read(chunk_size_bytes)
            if not raw_chunk:
                break

            text = carry + decoder.decode(raw_chunk, final=False)
            pretokens, carry = _extract_complete_pretokens(text, final=False)
            yield pretokens, len(raw_chunk), False

        tail = carry + decoder.decode(b"", final=True)
        if tail:
            pretokens, carry = _extract_complete_pretokens(tail, final=True)
            if carry:
                pretokens.append(carry)
            yield pretokens, 0, True


def _encode_batch(enc, batch: list[str]) -> list[int]:
    if not batch:
        return []
    # for gpt2
    # encoded_batch = enc.encode_ordinary_batch(batch)
    # flattened: list[int] = []
    # for ids in encoded_batch:
    #     flattened.extend(ids)

    flattened: list[int] = []
    for text in batch:
        encoded = enc.encode(text)
        flattened.extend(encoded)
    
    return flattened


def get_tokenizer():
    # for gpt2
    # enc = tiktoken.get_encoding("gpt2")
    # return enc
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tinystories_vocab_50257.txt",
        merges_filepath="data/tinystories_merges_50257.txt",
        special_tokens=["<|endoftext|>"]
        )
    return tokenizer


def encode_large_file(
    input_path: str,
    output_path: str,
    dtype: np.dtype,
    chunk_size_bytes: int,
    pretoken_batch_size: int,
    flush_tokens: int,
    log_interval_chunks: int,
) -> int:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_bytes = os.path.getsize(input_path)
    processed_bytes = 0
    processed_chunks = 0
    total_tokens = 0

    enc = get_tokenizer()
    pretoken_batch: list[str] = []
    token_buffer: list[int] = []

    with open(output_path, "wb") as out_f:
        for pretokens, chunk_bytes, _ in iter_pretokens_from_file(input_path, chunk_size_bytes):
            processed_chunks += 1
            processed_bytes += chunk_bytes

            pretoken_batch.extend(pretokens)
            while len(pretoken_batch) >= pretoken_batch_size:
                current = pretoken_batch[:pretoken_batch_size]
                del pretoken_batch[:pretoken_batch_size]
                token_buffer.extend(_encode_batch(enc, current))

                if len(token_buffer) >= flush_tokens:
                    arr = np.asarray(token_buffer, dtype=dtype)
                    arr.tofile(out_f)
                    total_tokens += int(arr.size)
                    token_buffer.clear()

            if processed_chunks % log_interval_chunks == 0:
                pct = 100.0 * processed_bytes / max(total_bytes, 1)
                print(
                    f"chunks={processed_chunks} bytes={processed_bytes}/{total_bytes} "
                    f"({pct:.2f}%) written_tokens={total_tokens}"
                )

        if pretoken_batch:
            token_buffer.extend(_encode_batch(enc, pretoken_batch))

        if token_buffer:
            arr = np.asarray(token_buffer, dtype=dtype)
            arr.tofile(out_f)
            total_tokens += int(arr.size)

    return total_tokens


def main() -> None:
    dtype = _dtype_from_name(CONFIG["dtype"])
    chunk_size_bytes = int(CONFIG["chunk_size_mb"]) * 1024 * 1024

    total_tokens = encode_large_file(
        input_path=CONFIG["input_path"],
        output_path=CONFIG["output_path"],
        dtype=dtype,
        chunk_size_bytes=chunk_size_bytes,
        pretoken_batch_size=int(CONFIG["pretoken_batch_size"]),
        flush_tokens=int(CONFIG["flush_tokens"]),
        log_interval_chunks=int(CONFIG["log_interval_chunks"]),
    )

    output_size = Path(CONFIG["output_path"]).stat().st_size
    print(f"Done. tokens={total_tokens} output_bytes={output_size} path={CONFIG['output_path']}")


if __name__ == "__main__":
    main("/root/autodl-tmp/TinyStoriesV2-GPT4-valid.txt", "data/mytokenizer/valid.bin")
    main("/root/autodl-tmp/TinyStoriesV2-GPT4-train.txt", "data/mytokenizer/train.bin")