import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# 1. Setup
device = "cuda"; dtype = torch.bfloat16
model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)

def get_emb(seq):
    protein = ESMProtein(sequence=seq)
    with torch.inference_mode():
        tokens = model.encode(protein).to(device)
        output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
        return output.embeddings.mean(dim=1).cpu().float().numpy().flatten()

# 2. Dataset: Real Proteins vs Junk
# Categorize them for the legend
data = {
    "Enzymes": ["MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCP", "MLPFLVAVGATTLFVLGVAA"],
    "Structural": ["MSYYHHHHHHADYDIPTTENLYFQGA", "MKTIIALSYIFCLVFADYKDDDDGAP"],
    "Junk": ["AAAAAAAAAAAAAAAAAAAA", "GSGSGSGSGSGSGSGSGSGS", "VLVLVLVLVLVLVLVLVLVL"]
}

embeddings = []
labels = []

for category, seqs in data.items():
    for seq in seqs:
        embeddings.append(get_emb(seq))
        labels.append(category)

# 3. UMAP Projection
reducer = umap.UMAP(n_neighbors=2, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(np.array(embeddings))

# 4. Plotting
plt.figure(figsize=(10, 7))
sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels, s=100)
plt.title("ESMC 600M: Unsupervised Protein Space Projection")
plt.savefig("protein_space.png")
print("Visual saved as protein_space.png")
