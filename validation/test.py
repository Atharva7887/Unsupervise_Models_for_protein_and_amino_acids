import torch
import os
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Force cache to the persistent volume
os.environ["HF_HOME"] = "/workspace/hf_cache"

def run_esmc_600m():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 

    print(f"--- Initializing ESMC 600M on {device} ---")
    
    try:
        # Loading the open-weights 600M model
        model = ESMC.from_pretrained("esmc_600m").to(device=device, dtype=dtype)
        print("Success: Model weights loaded.")

        seq = "MQIFVKTLTGKTITLEVEPSDTIEVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        protein = ESMProtein(sequence=seq)

        with torch.inference_mode():
            tokens = model.encode(protein).to(device)
            output = model.logits(
                tokens, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )

        print("\n--- RESULTS ---")
        print(f"Embedding Shape: {output.embeddings.shape}")
        # For 600M, the embedding dimension is 1152
        print(f"VRAM Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_esmc_600m()
