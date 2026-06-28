from cs336_basics.train_bpe import train_bpe
import os
import time
import json

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=5000,
        special_tokens=["<|endoftext|>"],
        num_proccesses = min(8, os.cpu_count() or 1)
    )
    end_time = time.time()
    print('train bpe took:', end_time - start_time)

    json_safe_vocab = {
        token_id: token_bytes.decode("utf-8", errors="replace")
        for token_id, token_bytes in vocab.items()
    }

    output_vocab_path = "tinystories_vocab.json"
    with open(output_vocab_path, "w", encoding="utf-8") as file:
        json.dump(json_safe_vocab, file, indent=4, ensure_ascii=False)
        
    print(f"Vocab successfully exported to {output_vocab_path}!")