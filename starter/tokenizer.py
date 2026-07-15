import os
import json
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocab_size = 1024
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.cache = {}
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.merges_path = os.path.join(current_dir, "merges.json")
        
        if os.path.exists(self.merges_path):
            self.load_merges()
        else:
            self.train_and_save()

    def train_and_save(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.abspath(os.path.join(current_dir, "..", "data", "train_corpus.txt"))
        if not os.path.exists(corpus_path):
            return
            
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read(500000)
            
        words = re.findall(r"\S+|\s+", text)
        word_freqs = defaultdict(int)
        for w in words:
            word_freqs[tuple(w.encode("utf-8"))] += 1
            
        for i in range(256, self.vocab_size):
            pair_counts = defaultdict(float)
            for word, freq in word_freqs.items():
                for pair in zip(word, word[1:]):
                    combined = self.vocab[pair[0]] + self.vocab[pair[1]]
                    is_deva = len(combined) >= 2 and combined[0] == 0xe0 and (combined[1] == 0xa4 or combined[1] == 0xa5)
                    weight = 2.0 if is_deva else 1.0
                    pair_counts[pair] += freq * weight
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges[best_pair] = i
            self.vocab[i] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            new_word_freqs = defaultdict(int)
            p0, p1 = best_pair
            for word, freq in word_freqs.items():
                if p0 not in word or p1 not in word:
                    new_word_freqs[word] += freq
                    continue
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and word[j] == p0 and word[j+1] == p1:
                        new_word.append(i)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freqs[tuple(new_word)] += freq
            word_freqs = new_word_freqs
            
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(self.merges_path, "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f)
            
        self.build_vocab()

    def load_merges(self):
        with open(self.merges_path, "r", encoding="utf-8") as f:
            serializable_merges = json.load(f)
        self.merges = {}
        for k, v in serializable_merges.items():
            a, b = map(int, k.split(","))
            self.merges[(a, b)] = v
        self.build_vocab()

    def build_vocab(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (a, b), idx in sorted_merges:
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    def encode(self, text):
        words = re.findall(r"\S+|\s+", text)
        encoded_ids = []
        for w in words:
            if w not in self.cache:
                ids = list(w.encode("utf-8"))
                while len(ids) >= 2:
                    pairs = set(zip(ids, ids[1:]))
                    valid_pairs = [p for p in pairs if p in self.merges]
                    if not valid_pairs:
                        break
                    best_pair = min(valid_pairs, key=lambda p: self.merges[p])
                    new_ids = []
                    i = 0
                    while i < len(ids):
                        if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                            new_ids.append(self.merges[best_pair])
                            i += 2
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    ids = new_ids
                self.cache[w] = ids
            encoded_ids.extend(self.cache[w])
        return encoded_ids

    def decode(self, ids):
        parts = []
        for token in ids:
            if token in self.vocab:
                parts.append(self.vocab[token])
        byte_str = b"".join(parts)
        return byte_str.decode("utf-8", errors="replace")

def load(path=None):
    return BPETokenizer()