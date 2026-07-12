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
        splits_by_spt = re.split(SPECIAL_PAT,content)
        for split in splits_by_spt:
            for c in re.finditer(PAT, split):           # split by PAT uisng finditer, an iterator over matches
                group = c.group().encode('utf-8')
                local_counts[tuple(group)] += 1         # tuple converts bytes to tuple of ints
    return local_counts

### PRETOKENIZE ###
def naive_pretokenize(input_path,special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    SPECIAL_PAT = "|".join(re.escape(spt) for spt in special_tokens)
    counts = defaultdict(int) # map tuples of pretokenized splits to frequencies
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()
        ### SPLIT BY SPECIAL TOKENS
        SPECIAL_PAT = "|".join(re.escape(spt) for spt in special_tokens)
        splits_by_spt = re.split(SPECIAL_PAT,content)
        for split in splits_by_spt:
            for c in re.finditer(PAT, split): # finditer is an iterator over matches
                group = c.group().encode('utf-8')
                counts[tuple(group)] += 1       # tuple converts bytes to tuple of ints
    return counts


def parallel_pretokenize(input_path,special_tokens,num_processes):
    """
        1. chunk the text by a special token, ensuring we don't split the corpus in the middle of a special token, we choose <|endoftext|>
        2. within each chunk, 
            - there may a special token given by the argument special_tokens in this chunk, so we also SPLIT by special tokens
            - only then, get a frequency count of the tuples given by SPLITTING by PAT 
        3. aggregate the local chunk counts into counts

        Why do we need special tokens? 
            1. we need to seperate documents so we don't merge tokens across documents as a merged token across documents has no meaning 
            2. tokens that should be atomic like <think> 
    """
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

    """
    Pretokenization is like a coarse-tokenization where we divide among natural boundaries like words, punctutation, numbers.
    so that when BPE is deciding what to merge, it only merges within these boundaries
    For example, hello world. would be pretokenized into ['hello',' world', .] 
    
    We would count co-occurences within each unit, rather than across. This prevents tokens like world., world!, world? as seperate vocabulary indexes. 
    """
    # for small files, no need for multiprocessing
    file_size = os.path.getsize(input_path) # get file size in bytes
    if file_size < 1_000_000:
        counts = naive_pretokenize(input_path,special_tokens)
    else:
        counts = parallel_pretokenize(input_path,special_tokens, num_processes)

    '''
    build
        1. co-occurences counts by counting over all byte pairs
        2. mapping between each cooccurence to what chunks it is in
        3. mapping of an id to a (chunk,chunk_freq)
    ''' 
    def get_stats(counts):
        cooccurence_counts = defaultdict(int)
        cooccurence_to_chunk_ids = defaultdict(set)
        id_to_chunk = defaultdict(list)
        for i, (key,count) in enumerate(counts.items()):
            id_to_chunk[i] = (key, count)
            for c1,c2 in zip(key,key[1:]):
                cooccurence_to_chunk_ids[(c1,c2)].add(i)
                cooccurence_counts[(c1,c2)]+=count
        return cooccurence_counts, cooccurence_to_chunk_ids, id_to_chunk
    
    def remove_counts(chunk_id):
        chunk, chunk_count = id_to_chunk[chunk_id]
        local_cooccurences = defaultdict(int)
        for c1,c2 in zip(chunk,chunk[1:]):
            local_cooccurences[(c1,c2)]+=1
        
        # for each co-occurence in this chunk, REMOVE how many times it occurs * how many times the chunk appears in the corpus to the global co-occurence counts
        for (c1,c2), local_freq in local_cooccurences.items():
            cooccurence_counts[(c1,c2)] -= local_freq * chunk_count
    
    def add_counts(chunk_id):
        chunk, chunk_count = id_to_chunk[chunk_id]
        local_cooccurences = defaultdict(int)
        for c1,c2 in zip(chunk,chunk[1:]):
            local_cooccurences[(c1,c2)]+=1
        
        # for each co-occurence in this chunk, ADD how many times it occurs * how many times the chunk appears in the corpus to the global co-occurence counts
        for (c1,c2), local_freq in local_cooccurences.items():
            cooccurence_counts[(c1,c2)] += local_freq * chunk_count
            cooccurence_to_chunk_ids[(c1,c2)].add(chunk_id)
        


    vocab = {idx : bytes([idx]) for idx in range(256)} # (vocab idx -> byte tuple) e.g 97-> b'a'
    merges = []
    iterations = vocab_size - 256 - len(special_tokens)
    new_vocab_idx = 256
    cooccurence_counts, cooccurence_to_chunk_ids, id_to_chunk = get_stats(counts) # get occurences one time only
    
    for i in range(iterations):
        # pick the max occurence and in case of tie, the lexiographically largest. very important you compare elementwise of the tuple and not the concatenated 
        top_pair = max(cooccurence_counts, key = lambda k: (cooccurence_counts[k], vocab[k[0]],vocab[k[1]]))
        # print(f'top pair for round {i} is {vocab[top_pair[0]] + vocab[top_pair[1]]} with count {cooccurence_counts[top_pair]}')
        # print(f'the number of chunks are related to this pair',len(cooccurence_to_chunk_ids[top_pair]))
        
        # adjust global counts by examining all chunks top_pair is in
        chunk_ids = list(cooccurence_to_chunk_ids[top_pair])
        for chunk_id in chunk_ids:
            chunk, chunk_count = id_to_chunk[chunk_id]
            # decrement the contribution to global count of ALL co-occurences in this chunk  
            remove_counts(chunk_id)

            # merge the top_pair co-occurence in this chunk
            new_chunk = []
            j = 0
            while j < len(chunk):
                if j+1 < len(chunk) and (chunk[j],chunk[j+1]) == top_pair:
                    new_chunk.append(new_vocab_idx)
                    j+=2
                else:
                    new_chunk.append(chunk[j])
                    j+=1
            id_to_chunk[chunk_id] = (tuple(new_chunk), chunk_count)

            # add back the contribution to global count of every co-occurences for the new merged chunk that now excludes top pair 
            add_counts(chunk_id)

        # top pair is no longer a cooccurence
        del cooccurence_counts[top_pair]
        del cooccurence_to_chunk_ids[top_pair]

        
        vocab[new_vocab_idx] = vocab[top_pair[0]] + vocab[top_pair[1]]  # concatenate the bytes
        merges.append((vocab[top_pair[0]],vocab[top_pair[1]]))
        
        new_vocab_idx +=1

    # add special tokens to vocabulary
    for i in range(len(special_tokens)):
        vocab[new_vocab_idx] = special_tokens[i].encode('utf-8')
        new_vocab_idx+=1
    
    return vocab, merges