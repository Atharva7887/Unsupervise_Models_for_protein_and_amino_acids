import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.metrics.pairwise import cosine_similarity

def extract_structural_pairs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print("Loading ESMC 600M...")
    model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)
    
    # WW Domain Sequence
    seq = "VPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK"
    protein = ESMProtein(sequence=seq)
    
    with torch.inference_mode():
        tokens = model.encode(protein).to(device)
        output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
        
        # THE FIX: Slice out the first (BOS) and last (EOS) tokens
        # We only want the embeddings that correspond exactly to the amino acids
        residue_embs = output.embeddings[0, 1:-1, :].cpu().float().numpy()
        
    # Calculate similarity matrix
    interaction_matrix = cosine_similarity(residue_embs)
    print(f"\nCorrected Matrix Shape: {interaction_matrix.shape} (Matches Sequence: {len(seq)}x{len(seq)})")

    # BIOLOGICAL INSIGHT: Extract top off-diagonal interactions
    # We ignore the diagonal (i=j) and adjacent residues (i=j+1, i=j-1) 
    # because covalent bonds are obvious. We want long-range folding interactions.
    
    # Create a mask to ignore the diagonal and immediate neighbors
    mask = np.ones(interaction_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    for i in range(len(seq) - 1):
        mask[i, i+1] = 0
        mask[i+1, i] = 0
        
    # Apply mask and find highest values
    masked_matrix = interaction_matrix * mask
    
    # Find the top 3 highest interacting pairs
    flat_indices = np.argsort(masked_matrix.flatten())[::-1]
    
    print("\n--- Top Predicted Long-Range Interactions ---")
    pairs_found = 0
    seen = set()
    
    for idx in flat_indices:
        i, j = divmod(idx, interaction_matrix.shape[1])
        # Avoid printing duplicates (i,j is the same as j,i)
        pair = tuple(sorted((i, j)))
        
        if pair not in seen and masked_matrix[i, j] > 0.5:
            seen.add(pair)
            print(f"Residue {i} ({seq[i]}) <--> Residue {j} ({seq[j]}) | Similarity: {masked_matrix[i, j]:.4f}")
            pairs_found += 1
            if pairs_found >= 3:
                break

if __name__ == "__main__":
    extract_structural_pairs()
