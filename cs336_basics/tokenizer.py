from collections.abc import Iterable, Iterator
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """

        self.vocab = vocab
        self.bpe_cache = {}
        self.special_tokens = special_tokens
        if self.special_tokens:
            special_tokens = sorted(special_tokens, key = lambda spt: -len(spt))
            self.SPECIAL_PAT = "|".join(re.escape(spt) for spt in special_tokens)
            self.SPECIAL_PAT = "("+self.SPECIAL_PAT+")"
        else:
            self.SPECIAL_PAT = ""

        self.bytes_to_vocab_idx = {} # map the byte object -> vocab idx
        for i, b in vocab.items():
            self.bytes_to_vocab_idx[b] = i
        
        # map the tuple of byte objects of the merge rule to the order learned, so we can find the cooccurence with the earliest rule during BPE
        self.merged_bytes_to_order = {} 
        for i in range(len(merges)):
            p1,p2 = merges[i] # p1, p2 are byte objects like b'h', b'e'
            self.merged_bytes_to_order[(p1,p2)] = i
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass
        

    def encode(self, text: str) -> list[int]:
        ### SPLIT BY special tokens
        if self.SPECIAL_PAT:
            splits_by_spt = re.split(self.SPECIAL_PAT,text)
        else:
            splits_by_spt = [text]
        
        def encode_chunk(chunk):
            byte_seq = chunk.encode('utf-8') # byte object
            if byte_seq in self.bpe_cache:
                return self.bpe_cache[byte_seq]

            token_level_bytes = [bytes([c]) for c in byte_seq] # separate out each byte; we need bytes() b/c iteration yields integers
            while len(token_level_bytes)>=2:
                cooccurences = set()
                for c1,c2 in zip(token_level_bytes,token_level_bytes[1:]):
                    cooccurences.add((c1,c2))
                # pick the cooccurence that has the earliest merge rule
                pair = min(cooccurences, key = lambda p : self.merged_bytes_to_order.get(p,float('inf')))
                # nothing else to merge
                if pair not in self.merged_bytes_to_order:
                    break

                # do the merge!
                new_token_level_bytes = []
                j = 0
                while j < len(token_level_bytes):
                    if j+1 < len(token_level_bytes) and (token_level_bytes[j],token_level_bytes[j+1]) == pair:
                        new_token_level_bytes.append(pair[0]+pair[1])
                        j+=2
                    else:
                        new_token_level_bytes.append(token_level_bytes[j])
                        j+=1
                token_level_bytes = new_token_level_bytes

            self.bpe_cache[byte_seq] = token_level_bytes
            return token_level_bytes

        encoded = []
        for split in splits_by_spt:
            if self.special_tokens and split in self.special_tokens:
                byte_seq = split.encode('utf-8')
                idx = self.bytes_to_vocab_idx[byte_seq]
                # print('spt',split,'encoded as',self.vocab[idx],'with idx',idx)
                encoded.append(idx)
            else:
                for chunk in re.findall(PAT, split): # finditer is an iterator over matches
                    bpe_merged = encode_chunk(chunk)
                    vocab_idxs = [self.bytes_to_vocab_idx[b] for b in bpe_merged]
                    encoded.extend(vocab_idxs)
        
        return encoded
        
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is equired for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        buf = ""
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids: list[int]) -> str:
        # concatenate all the bytes and THEN decode. we cannot decode each byte corresponding to each id b/c some unicode characters are more than one byte
        res = b"".join([self.vocab[id] for id in ids]).decode('utf-8',errors='replace')
        return res
        
    