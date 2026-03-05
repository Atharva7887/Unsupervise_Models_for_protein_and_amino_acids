import torch
import numpy as np
import pandas as pd
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)

def get_weighted_embedding(seq):
    protein = ESMProtein(sequence=seq)
    with torch.inference_mode():
        tokens = model.encode(protein).to(device)
        # We fetch embeddings AND logit outputs for better precision
        output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
        
        # INCREASE ACCURACY: Instead of a simple mean, we use the embedding 
        # of the first token (often the 'CLS' or global context token in Transformers)
        # which usually captures the most 'accurate' global representation.
        return output.embeddings[0, 0, :].cpu().float().numpy().reshape(1, -1)

# 2. Dataset for Accuracy Testing
# Two Hemoglobins (should be similar) vs One Green Fluorescent Protein (should be different)
test_data = {
    "Hemoglobin_A": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "Hemoglobin_B": "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH",
    "GFP_Fluorescent": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
}

# 3. Calculate Similarities
embeddings = {name: get_weighted_embedding(seq) for name, seq in test_data.items()}

# Compare Hemoglobin A vs B (Expecting > 0.85)
sim_aa_bb = cosine_similarity(embeddings["Hemoglobin_A"], embeddings["Hemoglobin_B"])[0][0]
# Compare Hemoglobin A vs GFP (Expecting < 0.60)
sim_aa_gfp = cosine_similarity(embeddings["Hemoglobin_A"], embeddings["GFP_Fluorescent"])[0][0]

print(f"--- Accuracy Report ---")
print(f"Similarity (Same Family: Hemo A vs B): {sim_aa_bb:.4f}")
print(f"Similarity (Different Family: Hemo vs GFP): {sim_aa_gfp:.4f}")

# Save to CSV for your documentation
df = pd.DataFrame([["Hemo A vs B", sim_aa_bb], ["Hemo vs GFP", sim_aa_gfp]], columns=["Comparison", "Score"])
df.to_csv("accuracy_report.csv", index=False)
