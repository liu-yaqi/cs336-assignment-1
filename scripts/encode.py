
import os
import yaml
import numpy as np
from cs336_basics.tokenizer import Tokenizer
import json
from transformers import GPT2Tokenizer


def encode_large_file(tokenizer, file_path, chunk_size=1024*1024, log_interval=10):
    ids = []
    total_bytes = os.path.getsize(file_path)
    processed_bytes = 0
    processed_chunks = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            encoded = tokenizer.encode(chunk)
            ids.extend(encoded)
            
            chunk_byte_len = len(chunk.encode('utf-8'))
            processed_bytes += chunk_byte_len
            processed_chunks += 1
            
            if processed_chunks % log_interval == 0:
                print(f"Chunk {processed_chunks}: processed {processed_bytes / 1024:.2f} KB, tokens: {len(ids)}")

    print(f"✅ Total tokens: {len(ids)}")
    return ids


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(input_path, out_path):
    # Rebuild tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print ("start tokenizer")

    text = "hello, world! I am your friend."
    ids = tokenizer.encode(text)
    print ("ids1:", ids)
    ids = np.array(ids, dtype=np.uint32)
    print ("ids2:", ids)

    ids = encode_large_file(tokenizer, input_path)
    ids = np.array(ids, dtype=np.uint32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ids.tofile(out_path)

    print(f"Saved {len(ids)} tokens to {out_path}")


if __name__ == "__main__":
    main()