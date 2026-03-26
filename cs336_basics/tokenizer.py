"""
train bpe tokenizer.
"""
import os
import regex
from collections import defaultdict
from pathlib import Path

import collections
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import Pool, cpu_count

SPLIT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids, freq=1, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + freq
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def process_chunk(chunk):
    """处理一个语料块"""
    local_freqs = collections.defaultdict(int)
    for m in regex.finditer(SPLIT_PAT, chunk):
        word = m[0]
        if not word:
            continue
        word_bytes = word.encode('utf-8')
        # tokens.extend(tuple(word_bytes))
        local_freqs[tuple(word_bytes)] += 1
    return local_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    assert vocab_size >= 256

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. 先用special_tokens分割文本
    if special_tokens:
        pattern = "|".join(map(regex.escape, sorted(special_tokens, key=len, reverse=True)))
        chunks = regex.split(f"({pattern})", text)
    else:
        chunks = [text]

    # 2. 对每个chunk用SPLIT_PAT分割，构建频率字典
    combined_freq = defaultdict(int)
    for chunk in chunks:
        if not chunk:  # 跳过空字符串
            continue
        # 检查是否是special token
        if special_tokens and chunk in special_tokens:
            # combined_freq[tuple(chunk.encode('utf-8'))] += 1
            pass
        else:
            # 用SPLIT_PAT分割普通文本
            for m in regex.finditer(SPLIT_PAT, chunk):
                word = m[0]
                if word:  # 跳过空匹配
                    combined_freq[tuple(word.encode('utf-8'))] += 1

    # 3. 初始化词表
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i, token in enumerate(special_tokens):
        vocab[255 + i] = token.encode('utf-8')
        
    # 4. 迭代合并
    merges = []
    merge_start_id = len(vocab)
    num_merges = vocab_size - len(vocab)
    
    # iteratively merge the most common pairs to create new tokens
    for i in range(num_merges):
        # count up the number of times every consecutive pair appears
        stats = {}
        for ids, freq in combined_freq.items():
            if len(ids) < 2:
                continue
            stats = get_stats(ids, freq, stats)
        
        # 如果没有可合并的对，提前退出
        if not stats:
            break
        
        # find the pair with the highest count
        pair = min(stats, key=lambda k: (-1.0 * stats.get(k, 0), (vocab[k[0]], vocab[k[1]])))
        # mint a new token: assign it the next available id
        idx = merge_start_id + i
        # replace all occurrences of pair in ids with idx
        new_combined = defaultdict(int)
        for ids, freq in combined_freq.items():
            if len(ids) < 2:
                new_combined[ids] += freq
                continue
            merged_ids = tuple(merge(ids, pair, idx))
            new_combined[merged_ids] += freq

        combined_freq = new_combined

        # save the merge
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    return vocab, merges


def fast_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    assert vocab_size >= 256

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. 先用special_tokens分割文本
    if special_tokens:
        pattern = "|".join(map(regex.escape, sorted(special_tokens, key=len, reverse=True)))
        chunks = regex.split(f"({pattern})", text)
    else:
        chunks = [text]

    # 2. 对每个chunk用SPLIT_PAT分割，构建频率字典
    # 使用进程池并行处理（chunks可以分割一下）
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_chunk, chunks)

    # 合并结果（注意内存效率）
    combined_freq = defaultdict(int)
    for idx, freqs in enumerate(results):
        # 合并频率统计
        for ids, count in freqs.items():
            combined_freq[ids] += count

    # 3. 初始化词表
    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i, token in enumerate(special_tokens):
        vocab[255 + i] = token.encode('utf-8')
        
    # 4. 迭代合并
    merges = []
    merge_start_id = len(vocab)
    num_merges = vocab_size - len(vocab)
    print('num_merges', num_merges)
    
    # iteratively merge the most common pairs to create new tokens
    for i in range(num_merges):
        # count up the number of times every consecutive pair appears
        stats = {}
        for ids, freq in combined_freq.items():
            if len(ids) < 2:
                continue
            stats = get_stats(ids, freq, stats)
        
        # 如果没有可合并的对，提前退出
        if not stats:
            break
        
        # find the pair with the highest count
        pair = max(stats, key=lambda k: (stats.get(k, 0), (vocab[k[0]], vocab[k[1]])))
        # mint a new token: assign it the next available id
        idx = merge_start_id + i
        # replace all occurrences of pair in ids with idx
        new_combined = defaultdict(int)
        for ids, freq in combined_freq.items():
            if len(ids) < 2:
                new_combined[ids] += freq
                continue
            merged_ids = tuple(merge(ids, pair, idx))
            new_combined[merged_ids] += freq

        combined_freq = new_combined

        # save the merge
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    return vocab, merges


class BPETrainer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.heap = []  # 最大堆（存储负频率）
        self.pair_freqs = defaultdict(int)  # (sym1, sym2) -> freq
        self.wordid_freqs = defaultdict(int)  # 记录每个单词的频率
        self.pair_to_wordid = defaultdict(set)  # 
        self.vocab = None
        self.merges = None

    def _update_heap(self, pair, freq):
        """
        更新堆中的 pair 频率。
        """
        heapq.heappush(self.heap, (-freq, pair))  # 使用负频率构建最大堆

    def get_max_pair(self):
        """
        获取当前频率最高的 pair。
        如果频率相同，返回字典序最大的 pair。
        """
        while self.heap:
            freq, pair = heapq.heappop(self.heap)
            freq = -freq  # 恢复正频率
            if self.pair_freqs.get(pair, 0) == freq:
                return pair, freq
        return None, 0

    def train(self, input_path, special_tokens):
        """
        训练 BPE 模型。
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. 初始化词表
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i, token in enumerate(special_tokens or []):
            vocab[255 + i] = token.encode('utf-8')
        
        # 2. 文本分割
        if special_tokens:
            pattern = "|".join(map(regex.escape, sorted(special_tokens, key=len, reverse=True)))
            chunks = regex.split(f"({pattern})", text)
        else:
            chunks = [text]

        
        # with open(..., "rb") as f:
        #     num_processes = 4
        #     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        #     # The following is a serial implementation, but you can parallelize this
        #     # by sending each start/end pair to a set of processes.
        #     for start, end in zip(boundaries[:-1], boundaries[1:]):
        #         f.seek(start)
        #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #         # Run pre-tokenization on your chunk and store the counts for each pre-token

        # 3. pretoken and 构建频率字典
        for chunk in chunks:
            if not chunk or (special_tokens and chunk in special_tokens):
                continue
            else:
                # 用SPLIT_PAT分割普通文本
                for m in regex.finditer(SPLIT_PAT, chunk):
                    word = m[0]
                    if word:  # 跳过空匹配
                        ids = tuple(word.encode('utf-8'))
                        self.wordid_freqs[ids] += 1

        for ids, freq in self.wordid_freqs.items():
            if len(ids) < 2:
                continue
            for pair in zip(ids, ids[1:]): # iterate consecutive elements
                self.pair_freqs[pair] += freq
                self.pair_to_wordid[pair].add(ids)

        # 4. 初始化堆
        for pair, freq in self.pair_freqs.items():
            heapq.heappush(self.heap, (-freq, pair))

        # 5. merge 开始 BPE 训练
        merges = []
        merge_start_id = len(vocab)
        for idx in range(merge_start_id, self.vocab_size):
            # 获取频率最高的 pair
            pair, freq = self.get_max_pair()
            # pair = min(self.pair_freqs, key=lambda k: (-1.0 * self.pair_freqs.get(k, 0), (vocab[k[0]], vocab[k[1]])))
            if not pair or freq == 0:
                break

            # 创建新 token
            new_token = vocab[pair[0]] + vocab[pair[1]]
            vocab[idx] = new_token
            merges.append((vocab[pair[0]], vocab[pair[1]]))

            # 更新单词和 pair信息
            affected_wordids = self.pair_to_wordid[pair]
            del self.pair_to_wordid[pair]
            del self.pair_freqs[pair]
            for wordid in affected_wordids:
                freq = self.wordid_freqs[wordid]
                del self.wordid_freqs[wordid]

                new_wordid, add_pairs, del_pairs = self.merge_and_update_pairs(wordid, pair, idx)

                self.wordid_freqs[new_wordid] = freq

                for p in zip(new_wordid, new_wordid[1:]):
                    self.pair_to_wordid[p].discard(wordid)
                    self.pair_to_wordid[p].add(new_wordid)

                for p, v in add_pairs.items():
                    self.pair_freqs[p] += freq * v
                    self._update_heap(p, self.pair_freqs[p])
                for p, v in del_pairs.items():
                    self.pair_to_wordid[p].discard(wordid)
                    self.pair_freqs[p] -= freq * v
                    self._update_heap(p, self.pair_freqs[p])

        self.vocab = vocab
        self.merges = merges
        return vocab, merges

    def save(self, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike) -> None:
        """保存训练结果，文件格式与 Tokenizer.from_files 兼容。"""
        if self.vocab is None or self.merges is None:
            raise RuntimeError("Please call train() before save().")

        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        merges_path.parent.mkdir(parents=True, exist_ok=True)

        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token_id in sorted(self.vocab.keys()):
                token_bytes = self.vocab[token_id]
                f.write(f"{token_id}\t{token_bytes.hex()}\n")

        with open(merges_path, 'w', encoding='utf-8') as f:
            for token1, token2 in self.merges:
                f.write(f"{token1.hex()}\t{token2.hex()}\n")
    
    def merge_and_update_pairs(self, ids, pair, idx):
        newids = []
        del_pairs = defaultdict(int)
        add_pairs = defaultdict(int)
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                if i>0:
                    if i >= 2 and (ids[i-2], ids[i-1]) == (pair[0], pair[1]):
                        pass
                    else:
                        del_pairs[tuple([ids[i-1], pair[0]])] += 1
                if i+2 < len(ids):
                    del_pairs[tuple([pair[1], ids[i+2]])] += 1
                i += 2
            else:
                newids.append(ids[i])
                i += 1

        for id1, id2 in zip(newids, newids[1:]):
            if id1 == idx or id2 == idx:
                add_pairs[tuple([id1, id2])] += 1

        return tuple(newids), add_pairs, del_pairs
    

class BPETrainer1:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.heap = []  # 最大堆（存储负频率）
        self.pair_freqs = defaultdict(int)  # (sym1, sym2) -> freq
        self.word_freqs = {}  # 记录每个单词的频率
        self.pair_to_wordind = defaultdict(set)  # 
        self.merges = []
        self.vocab = {}

    def _update_heap(self, pair, freq):
        """
        更新堆中的 pair 频率。
        """
        heapq.heappush(self.heap, (-freq, pair)) # 使用负频率构建最大堆

    def get_max_pair(self):
        """
        获取当前频率最高的 pair。
        如果频率相同，返回字典序最大的 pair。
        """
        while self.heap:
            freq, pair = heapq.heappop(self.heap)
            freq = -freq  # 恢复正频率
            if self.pair_freqs.get(pair, 0) == freq:
                return pair, freq
        return None, 0

    def train(self, input_path, special_tokens):
        """
        训练 BPE 模型。
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. 初始化词表
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for i, token in enumerate(special_tokens or []):
            self.vocab[255 + i] = token.encode('utf-8')
        
        # 2. 文本分割
        if special_tokens:
            pattern = "|".join(map(regex.escape, sorted(special_tokens, key=len, reverse=True)))
            chunks = regex.split(f"({pattern})", text)
        else:
            chunks = [text]

        # 3. pretoken and 构建频率字典
        tmp_wordid_freqs = defaultdict(int)
        for chunk in chunks:
            if not chunk or (special_tokens and chunk in special_tokens):
                continue
            else:
                # 用SPLIT_PAT分割普通文本
                for m in regex.finditer(SPLIT_PAT, chunk):
                    word = m[0]
                    if word:  # 跳过空匹配
                        ids = tuple(word.encode('utf-8'))
                        tmp_wordid_freqs[ids] += 1
        word_ind = 0
        for ids, freq in tmp_wordid_freqs.items():
            self.word_freqs[word_ind] = (ids, freq)
            word_ind += 1
            if len(ids) < 2:
                continue
            for pair in zip(ids, ids[1:]): # iterate consecutive elements
                self.pair_freqs[pair] += freq
                self.pair_to_wordind[pair].add(word_ind - 1)

        # 4. 初始化堆
        for pair, freq in self.pair_freqs.items():
            heapq.heappush(self.heap, (-freq, pair))

        # 5. merge 开始 BPE 训练
        merge_start_id = len(self.vocab)
        for idx in range(merge_start_id, self.vocab_size):
            # 获取频率最高的 pair
            pair, freq = self.get_max_pair()
            # pair1 = min(self.pair_freqs, key=lambda k: (-1.0 * self.pair_freqs.get(k, 0), (vocab[k[0]], vocab[k[1]])))

            if not pair or freq == 0:
                break

            # 创建新 token
            new_token = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[idx] = new_token
            self.merges.append((self.vocab[pair[0]], self.vocab[pair[1]]))

            # 更新单词和 pair信息
            affected_word_inds = self.pair_to_wordind[pair]
            del self.pair_to_wordind[pair]
            del self.pair_freqs[pair]
            for word_ind in affected_word_inds:
                wordid, freq = self.word_freqs[word_ind]

                new_wordid, add_pairs, del_pairs, del_pairs_flag = self.merge_and_update_pairs(wordid, pair, idx)

                self.word_freqs[word_ind] = (new_wordid, freq)

                for p, v in add_pairs.items():
                    self.pair_to_wordind[p].add(word_ind)
                    self.pair_freqs[p] += freq * v
                    self._update_heap(p, self.pair_freqs[p])
                for p, v in del_pairs.items():
                    if del_pairs_flag[p]:
                        self.pair_to_wordind[p].discard(word_ind)
                    self.pair_freqs[p] -= freq * v
                    self._update_heap(p, self.pair_freqs[p])

        return self.vocab, self.merges
    
    def merge_and_update_pairs(self, ids, pair, idx):
        newids = []
        del_pairs = defaultdict(int)
        add_pairs = defaultdict(int)
        del_pairs_discarded = defaultdict(bool)
        i = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                if i>0:
                    if i >= 2 and (ids[i-2], ids[i-1]) == (pair[0], pair[1]):
                        pass
                    else:
                        del_pairs[tuple([ids[i-1], pair[0]])] += 1
                        del_pairs_discarded[tuple([ids[i-1], pair[0]])] = True
                if i+2 < len(ids):
                    del_pairs[tuple([pair[1], ids[i+2]])] += 1
                    del_pairs_discarded[tuple([pair[1], ids[i+2]])] = True
                i += 2
            else:
                newids.append(ids[i])
                i += 1

        for id1, id2 in zip(newids, newids[1:]):
            if id1 == idx or id2 == idx:
                add_pairs[tuple([id1, id2])] += 1
            if (id1, id2) in del_pairs_discarded:
                del_pairs_discarded[(id1, id2)] = False

        return tuple(newids), add_pairs, del_pairs, del_pairs_discarded


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        # 构建反向索引：bytes -> token_id（用于encode）
        self.dstoi= {v: k for k, v in vocab.items()}

        # 构建merges的反向索引： (token_id1, token_id2) -> merge_order_index
        self.merges_ind = {}
        for i, pair in enumerate(merges):
            self.merges_ind[(self.dstoi[pair[0]], self.dstoi[pair[1]])] = i

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为token IDs列表
        
        步骤：
        1. 先用special_tokens分割文本
        2. 对非special部分用SPLIT_PAT分割成word
        3. 将每个word转为字节，逐个字节转为初始token ID（0-255）
        4. 按merges顺序应用BPE合并
        """
        # 1. 先用special_tokens分割
        if self.special_tokens:
            pattern = "|".join(map(regex.escape, sorted(self.special_tokens, key=len, reverse=True)))
            chunks = regex.split(f"({pattern})", text)
        else:
            chunks = [text]

        ids = []
        # 2. 处理每个chunk
        for chunk in chunks:
            if not chunk:
                continue

            # 检查是否是special token
            if self.special_tokens and chunk in self.special_tokens:
                # special token直接查反向表
                token_bytes = chunk.encode('utf-8')
                ids.append(self.dstoi[token_bytes])
            else:
                # 用SPLIT_PAT分割普通文本
                for m in regex.finditer(SPLIT_PAT, chunk):
                    word = m[0]
                    if not word:
                        continue                    
                    # 3. 将word转为字节列表，每个字节作为初始token ID
                    word_bytes = word.encode('utf-8')
                    word_bytes = [bytes([b]) for b in list(word_bytes)]
                    word_ids = [self.dstoi[b] for b in word_bytes]
                    
                    # 4. 按merges顺序应用BPE合并
                    while len(word_ids)>=2:
                        pairs = set(zip(word_ids, word_ids[1:]))
                        pair = min(pairs, key=lambda p: self.merges_ind.get(p, float("inf")))
                        if pair not in self.merges_ind:
                            break
                        idx = self.dstoi[self.vocab[pair[0]]+self.vocab[pair[1]]]
                        word_ids = merge(word_ids, pair, idx)
                    ids.extend(word_ids)
        return ids
    
    def decode(self, ids: list[int]) -> str:
        """
        将token IDs列表解码为文本
        
        Args:
            ids (list[int]): Token IDs列表
            
        Returns:
            str: 解码后的文本
        """
        token_bytes = b"".join(self.vocab.get(id, b"") for id in ids)
        text = token_bytes.decode('utf-8', errors='replace')
        return text
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_id, token_bytes = line.split('\t')
                vocab[int(token_id)] = bytes.fromhex(token_bytes)
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token1, token2 = line.split('\t')
                merges.append((bytes.fromhex(token1), bytes.fromhex(token2)))
        return cls(vocab, merges, special_tokens)

    def encode_iterable(self, iterable):
        for text_chunk in iterable:
            # 编码当前文本块为 token IDs
            tokens = self.encode(text_chunk)
            yield tokens


if __name__ == "__main__":

    import sys
    sys.path.append('assignment1-basics-main/tests')

    from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
    import json
    import time
    import cProfile
    from io import StringIO
    import pstats

    input_path = FIXTURES_PATH / "corpus.en"

    # -------测试运行时间1
    trainer = BPETrainer(500)
    cProfile.run('vocab1, merges1 = trainer.train(input_path, ["<|endoftext|>"])', sort='cumulative')
    # cProfile.run('vocab, merges = train_bpe(input_path, 500, special_tokens=["<|endoftext|>"])', sort='cumulative')

    # -------测试运行时间2
    # pr = cProfile.Profile()
    # pr.enable()  # 开始分析

    # trainer = BPETrainer(500)
    # vocab1, merges1 = trainer.train(input_path, ["<|endoftext|>"])

    # pr.disable()  # 结束分析

    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats()
    # print(s.getvalue())


    # ---对比
    # start = time.time()
    # trainer = BPETrainer(500)
    # vocab, merges = trainer.train(input_path, ["<|endoftext|>"])
    # print(time.time() - start)

    # start = time.time()
    # trainer1 = BPETrainer1(500)
    # vocab1, merges1 = trainer1.train(input_path, ["<|endoftext|>"])
    # print(time.time() - start)

    # start = time.time()
    # vocab2, merges2 = train_bpe(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # print(time.time() - start)

    # print(len(merges2))
    # print(len(merges))
    # print(len(merges1))

    # assert merges1 == merges2
    # assert merges == merges1

    # for a,b in zip(merges, merges1):
    #     print(a,b)

    # assert merges == merges1



    # start = time.time()
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # print(time.time() - start)

    # for a, b in zip(merges, merges1):
    #     if a!=b:
    #         print(a, b)

    # assert merges1 == merges

    # reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    # reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # # Compare the learned merges to the expected output merges
    # gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    # with open(reference_merges_path, encoding="utf-8") as f:
    #     gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
    #     reference_merges = [
    #         (
    #             bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
    #             bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    #         )
    #         for merge_token_1, merge_token_2 in gpt2_reference_merges
    #     ]

    # # Compare the vocab to the expected output vocab
    # with open(reference_vocab_path, encoding="utf-8") as f:
    #     gpt2_reference_vocab = json.load(f)
    #     reference_vocab = {
    #         gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
    #         for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
    #     }

    # print(set(vocab.values()).difference(set(reference_vocab.values())))
    # print(len(vocab), len(reference_vocab))
    # print(len(merges), len(reference_merges))

    #  # Rather than checking that the vocabs exactly match (since they could

    # for a, b in zip(merges, reference_merges):
    #     print(a, b)


    # assert merges == reference_merges
    # assert set(vocab.keys()) == set(reference_vocab.keys())
    # assert set(vocab.values()) == set(reference_vocab.values())
