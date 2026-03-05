import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.metrics.pairwise import cosine_similarity

def generate_topology_map():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 # Optimized for RTX A4500
    
    print(f"Loading ESMC 600M on {device}...")
    try:
        model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)
        
        # Using a well-known structured protein (e.g., a small domain)
        # Sequence: WW Domain (folds into a clear beta-sheet structure)
        seq = "VPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK"
        protein = ESMProtein(sequence=seq)
        
        with torch.inference_mode():
            tokens = model.encode(protein).to(device)
            output = model.logits(
                tokens, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            
            # Extract the L x 1152 matrix (ignoring BOS/EOS tokens if present)
            # Assuming output.embeddings shape is [1, L, 1152]
            residue_embeddings = output.embeddings[0].cpu().float().numpy()
            
        # Mathematical core: Calculate L x L pairwise interaction matrix
        # This acts as an unsupervised proxy for 3D contact/distance maps
        interaction_matrix = cosine_similarity(residue_embeddings)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            interaction_matrix, 
            cmap="viridis", 
            square=True, 
            cbar_kws={'label': 'Residue-Residue Similarity'}
        )
        plt.title("Unsupervised 2D Topology Map (ESMC 600M)")
        plt.xlabel("Residue Index")
        plt.ylabel("Residue Index")
        
        file_name = "topology_map.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"\nSuccess: 2D Interaction map saved as {file_name}")
        print(f"Matrix Shape: {interaction_matrix.shape} (Expected: {len(seq)}x{len(seq)})")

    except Exception as e:
        print(f"Pipeline Error: {e}")
    finally:
        # VRAM Management for continuous pipeline running
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    generate_topology_map()
