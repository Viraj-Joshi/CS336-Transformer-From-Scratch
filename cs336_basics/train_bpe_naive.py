from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


import json
import regex as re
from collections import defaultdict

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

from multiprocessing import Pool

def process_chunk(
    input_path,
    special_tokens,
    start:int,
    end:int
):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    SPECIAL_PAT = "|".join(re.escape(spt) for spt in special_tokens)

    local_counts = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        content = f.read(end - start).decode("utf-8", errors="ignore")
        ### SPLIT BY SPECIAL TOKENS that could still be in the chunk
        chunks = re.split(SPECIAL_PAT,content)
        for chunk in chunks:
            for c in re.finditer(PAT, chunk):           # finditer is an iterator over matches
                group = c.group().encode('utf-8')
                local_counts[tuple(group)] += 1         # tuple converts bytes to tuple of ints
    return local_counts

### PRETOKENIZE ###
def naive_pretokenize(input_path, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    SPECIAL_PAT = "|".join(re.escape(spt) for spt in special_tokens)
    counts = defaultdict(int) # map tuples of pretokenized chunks to frequencies
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()
        ### SPLIT BY SPECIAL TOKENS
        SPECIAL_PAT = "|".join(re.escape(spt) for spt in special_tokens)
        chunks = re.split(SPECIAL_PAT,content)
        for chunk in chunks:
            for c in re.finditer(PAT, chunk): # finditer is an iterator over matches
                group = c.group().encode('utf-8')
                counts[tuple(group)] += 1       # tuple converts bytes to tuple of ints
    return counts


def parallel_pretokenize(input_path,special_tokens,num_processes):
    counts = defaultdict(int) # global map tuples of pretokenized chunks to frequencies

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    with Pool(num_processes) as pool:
        results = pool.starmap(process_chunk, [(input_path, special_tokens, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])])
        for r in results:
            for k, v in r.items():
                counts[k] += v 
    return counts

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    
    # for small files, no need for multiprocessing
    file_size = os.path.getsize(input_path) # get file size in bytes
    if file_size < 1_000_000:
        counts = naive_pretokenize(input_path, special_tokens)
    else:
        counts = parallel_pretokenize(input_path,special_tokens, num_processes)

    ### BPE over each pretokenized chunks (different than splitting by special tokens)

    # get co-occurences by counting over all byte pairs
    def get_stats(counts):
        aggregated_cooccurences = defaultdict(int)
        for key,count in counts.items():
            for c1,c2 in zip(key,key[1:]):
                aggregated_cooccurences[(c1,c2)]+=count
        return aggregated_cooccurences

    def merge_counts(counts, target_pair, vocab_idx):
        new_counts = {}
        for chunk, count in counts.items():
            new_chunk = []
            i = 0
            while i < len(chunk):
                if i+1 < len(chunk) and (chunk[i],chunk[i+1]) == target_pair:
                    new_chunk.append(vocab_idx)
                    i+=2
                else:
                    new_chunk.append(chunk[i])
                    i+=1
            new_counts[tuple(new_chunk)] = count
        # print(new_counts)
        return new_counts

    vocab = {idx : bytes([idx]) for idx in range(256)} # (vocab idx -> byte tuple)
    merges = []
    iterations = vocab_size - 256 - len(special_tokens)
    new_vocab_idx = 256
    for i in range(iterations):
        aggregated_cooccurences = get_stats(counts)
        # pick the max occurence and in case of tie, the lexiographically largest. very important you compare elementwise of the tuple and not the concatenated 
        top_pair = max(aggregated_cooccurences, key = lambda k: (aggregated_cooccurences[k], vocab[k[0]],vocab[k[1]]))
        # print(f'top pair for round {i} is {vocab[top_pair[0]] + vocab[top_pair[1]]} with count {aggregated_cooccurences[top_pair]}')
        
        vocab[new_vocab_idx] = vocab[top_pair[0]] + vocab[top_pair[1]]  # concatenate the bytes
        merges.append((vocab[top_pair[0]],vocab[top_pair[1]]))
        
        counts = merge_counts(counts,top_pair,new_vocab_idx)
        new_vocab_idx +=1

    # add special tokens to vocabulary
    for i in range(len(special_tokens)):
        vocab[new_vocab_idx] = special_tokens[i].encode('utf-8')
        new_vocab_idx+=1
    
    return vocab, merges