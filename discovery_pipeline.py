import os
# Mandatory cache routing for cloud containers
os.environ["HF_HOME"] = "/workspace/hf_cache"

import torch
import gc
import numpy as np
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, EsmForProteinFolding

class UnsupervisedProteinPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Pipeline on {self.device}...")
        
    def extract_top_contacts(self, sequence, threshold=0.6):
        """
        Phase 1: Uses ESMC 600M to predict 2D spatial contacts unsupervised.
        """
        print("\n--- Phase 1: ESMC 2D Contact Prediction ---")
        # Load in bfloat16 for efficiency
        model = ESMC.from_pretrained("esmc_600m").to(device=self.device, dtype=torch.bfloat16)
        protein = ESMProtein(sequence=sequence)
        
        with torch.inference_mode():
            tokens = model.encode(protein).to(self.device)
            output = model.logits(tokens, LogitsConfig(sequence=True, return_embeddings=True))
            # Slice out <BOS> and <EOS> tokens
            residue_embs = output.embeddings[0, 1:-1, :].cpu().float().numpy()
            
        interaction_matrix = cosine_similarity(residue_embs)
        
        # Mask diagonal and immediate neighbors
        mask = np.ones(interaction_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        for i in range(len(sequence) - 1):
            mask[i, i+1] = 0
            mask[i+1, i] = 0
            
        masked_matrix = interaction_matrix * mask
        flat_indices = np.argsort(masked_matrix.flatten())[::-1]
        
        print("Top Predicted Long-Range Interactions:")
        seen = set()
        count = 0
        for idx in flat_indices:
            i, j = divmod(idx, interaction_matrix.shape[1])
            pair = tuple(sorted((i, j)))
            
            if pair not in seen and masked_matrix[i, j] > threshold:
                seen.add(pair)
                print(f"  Residue {i} ({sequence[i]}) <--> Residue {j} ({sequence[j]}) | Score: {masked_matrix[i, j]:.4f}")
                count += 1
                if count >= 3: break
                
        # VRAM Cleanup before loading the massive folding model
        del model
        torch.cuda.empty_cache()
        gc.collect()

    def generate_3d_structure(self, sequence, output_filename="predicted_structure.pdb"):
        """
        Phase 2: Uses ESMFold to generate atomic coordinates.
        """
        print("\n--- Phase 2: ESMFold 3D Generation ---")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        # Must load in FP32 to avoid Hugging Face openfold loss bugs
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(self.device)
        
        if len(sequence) > 64:
            model.trunk.set_chunk_size(64)
            print("  [Optimization] Memory chunking enabled.")
            
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to(self.device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            pdb_string = model.output_to_pdb(outputs)[0]
            plddt_score = outputs.plddt.mean().item() * 100 
            
        with open(output_filename, "w") as f:
            f.write(pdb_string)
            
        print(f"  Structure saved to: {output_filename}")
        print(f"  Overall Confidence (pLDDT): {plddt_score:.2f} / 100")
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    # The WW Domain Sequence
    target_sequence = "VPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK"
    
    pipeline = UnsupervisedProteinPipeline()
    
    # Run the full suite
    pipeline.extract_top_contacts(target_sequence)
    pipeline.generate_3d_structure(target_sequence, "ww_domain_final.pdb")
