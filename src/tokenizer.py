class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # lookup tables
        self.ch2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2ch = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.ch2idx[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return "".join([self.idx2ch[i] for i in indices])
