import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. Setup
device = "cuda"; dtype = torch.bfloat16
model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)

def get_clean_embedding(seq):
    protein = ESMProtein(sequence=seq)
    with torch.inference_mode():
        tokens = model.encode(protein).to(device)
        output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
        # REMOVE PADDING: Only take the actual sequence length
        emb = output.embeddings[0, 1:len(seq)+1, :].cpu().float().numpy()
        # MEAN POOLING + TRANSPOSE
        return np.mean(emb, axis=0).reshape(1, -1)

# 2. Dataset
test_data = {
    "Hemo_A": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "Hemo_B": "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    "GFP": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Insulin": "GIVEQCCTSICSLYQLENYCN"
}

# 3. Generate and Scale Embeddings
raw_embs = np.array([get_clean_embedding(s) for s in test_data.values()]).squeeze()

# ACCURACY FIX: StandardScaler removes the 'background noise' common to all proteins
scaler = StandardScaler()
scaled_embs = scaler.fit_transform(raw_embs)

# 4. Create Similarity Matrix
sim_matrix = cosine_similarity(scaled_embs)
names = list(test_data.keys())

# 5. Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(sim_matrix, annot=True, xticklabels=names, yticklabels=names, cmap="YlGnBu")
plt.title("High-Resolution Protein Similarity (Standardized)")
plt.savefig("accuracy_heatmap.png")
print("New Heatmap saved as accuracy_heatmap.png")

# Report results
print(f"Hemo A vs B Similarity: {sim_matrix[0,1]:.4f}")
print(f"Hemo A vs GFP Similarity: {sim_matrix[0,2]:.4f}")
