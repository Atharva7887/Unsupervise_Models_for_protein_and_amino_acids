import os
# Mandatory cache routing
os.environ["HF_HOME"] = "/workspace/hf_cache"

import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, EsmForProteinFolding

class UnsupervisedProteinPipeline:
    def __init__(self, base_dir="/workspace/ESMC"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_dir = base_dir
        
        # Define output routes
        self.plots_dir = os.path.join(self.base_dir, "results", "plots")
        self.structures_dir = os.path.join(self.base_dir, "results", "structures")
        
        # Ensure directories exist just in case
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.structures_dir, exist_ok=True)
        
        print(f"Initializing Production Pipeline on {self.device}...")
        
    def extract_and_plot_contacts(self, sequence, protein_name="target"):
        """Phase 1: 2D Contact Prediction & Visualization"""
        print(f"\n--- Phase 1: ESMC 2D Contact Prediction ({protein_name}) ---")
        model = ESMC.from_pretrained("esmc_600m").to(device=self.device, dtype=torch.bfloat16)
        protein = ESMProtein(sequence=sequence)
        
        with torch.inference_mode():
            tokens = model.encode(protein).to(self.device)
            output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
            residue_embs = output.embeddings[0, 1:-1, :].cpu().float().numpy()
            
        interaction_matrix = cosine_similarity(residue_embs)
        
        # Mask diagonal and immediate neighbors for logic extraction
        mask = np.ones(interaction_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        for i in range(len(sequence) - 1):
            mask[i, i+1] = 0
            mask[i+1, i] = 0
            
        masked_matrix = interaction_matrix * mask
        
        # Extract Top Pairs
        flat_indices = np.argsort(masked_matrix.flatten())[::-1]
        print("Top Predicted Long-Range Interactions:")
        seen = set()
        count = 0
        for idx in flat_indices:
            i, j = divmod(idx, interaction_matrix.shape[1])
            pair = tuple(sorted((i, j)))
            if pair not in seen and masked_matrix[i, j] > 0.6:
                seen.add(pair)
                print(f"  Residue {i} ({sequence[i]}) <--> Residue {j} ({sequence[j]}) | Score: {masked_matrix[i, j]:.4f}")
                count += 1
                if count >= 3: break

        # Generate and route the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(interaction_matrix, cmap="viridis", square=True)
        plt.title(f"2D Topology Map: {protein_name}")
        plot_path = os.path.join(self.plots_dir, f"{protein_name}_topology.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  [Artifact] Heatmap routed to: {plot_path}")
        plt.close()
                
        del model
        torch.cuda.empty_cache()
        gc.collect()

    def generate_3d_structure(self, sequence, protein_name="target"):
        """Phase 2: ESMFold 3D Generation"""
        print(f"\n--- Phase 2: ESMFold 3D Generation ({protein_name}) ---")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(self.device)
        
        if len(sequence) > 64:
            model.trunk.set_chunk_size(64)
            print("  [Optimization] Memory chunking enabled.")
            
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(self.device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            pdb_string = model.output_to_pdb(outputs)[0]
            plddt_score = outputs.plddt.mean().item() * 100 
            
        # Route the structure file
        structure_path = os.path.join(self.structures_dir, f"{protein_name}_predicted.pdb")
        with open(structure_path, "w") as f:
            f.write(pdb_string)
            
        print(f"  [Artifact] Structure routed to: {structure_path}")
        print(f"  Overall Confidence (pLDDT): {plddt_score:.2f} / 100")
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # Ensure you run this from the root /workspace/ESMC directory
    pipeline = UnsupervisedProteinPipeline()
    
    target_sequence = "VPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK"
    target_name = "WW_Domain"
    
    pipeline.extract_and_plot_contacts(target_sequence, target_name)
    pipeline.generate_3d_structure(target_sequence, target_name)
