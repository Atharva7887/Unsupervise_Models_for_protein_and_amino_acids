import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def get_embedding(model, sequence, device):
    protein = ESMProtein(sequence=sequence)
    with torch.inference_mode():
        tokens = model.encode(protein).to(device)
        output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
        # Mean pooling: Average across the length (dim 1) to get one vector per protein
        return output.embeddings.mean(dim=1).cpu().numpy()

# Load model once
device = "cuda"; dtype = torch.bfloat16
model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)

# Test sequences: Insulin, Ubiquitin, and a random junk sequence
sequences = {
    "Insulin": "GIVEQCCTSICSLYQLENYCN",
    "Ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIEVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "Junk_PolyA": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
}

for name, seq in sequences.items():
    emb = get_embedding(model, seq, device)
    print(f"{name} Embedding (first 5 values): {emb[0][:5]}")
