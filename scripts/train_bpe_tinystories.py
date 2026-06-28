from cs336_basics.train_bpe import train_bpe
import os
import time
import json
import pickle

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_processes = min(8, os.cpu_count() or 1)
    )
    end_time = time.time()
    print(f'train bpe took {end_time - start_time} seconds')

    # save output
    output_dir = '.'
    vocab_path = os.path.join(output_dir, "tinystories_bpe_vocab.pkl")
    merges_path = os.path.join(output_dir, "tinystories_bpe_merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # stats
    longest = max(vocab.items(), key = lambda e : len(e[1]))
    print(f'longest token id={longest[0]}')
    print(f'longest token is {longest[1].decode('utf-8',errors='replace')}')
    print(f'longest token length in bytes is {len(longest[1])}')
    print(f'longest token length as a string {len(longest[1].decode('utf-8',errors='replace'))}')

