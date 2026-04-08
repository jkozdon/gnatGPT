# test_data.py
from tokenizer import CharTokenizer
from dataset import make_dataloader

text = open("data/shakespeare.txt").read()
tokenizer = CharTokenizer(text)
print(f"vocab size: {tokenizer.vocab_size}")        # expect ~65
print(f"corpus length: {len(text):,} chars")

loader = make_dataloader(text, tokenizer, context_len=128, batch_size=32)

x, y = next(iter(loader))
print(f"x shape: {x.shape}")    # expect torch.Size([32, 128])
print(f"y shape: {y.shape}")    # expect torch.Size([32, 128])
print(f"x dtype: {x.dtype}")    # expect torch.int64

# round-trip check: decode first sequence in the batch
print(tokenizer.decode(x[0].tolist()[:50]))   # should be readable Shakespeare
