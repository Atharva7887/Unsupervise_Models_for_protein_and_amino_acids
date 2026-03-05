import os
# Force Hugging Face to use your 30GB persistent volume
os.environ["HF_HOME"] = "/workspace/hf_cache"

import torch
import gc
import traceback
from transformers import AutoTokenizer, EsmForProteinFolding

def fold_sequence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESMFold v1 on {device} in native FP32...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        # THE FIX: Removed torch_dtype=torch.float16
        # We are relying on your 20GB VRAM to tank the 12GB FP32 model footprint.
        # This completely bypasses the float16 NaN indexing bugs in Hugging Face.
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
        
        seq = "VPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRK"
        print(f"Folding sequence of length {len(seq)}...")
        
        if len(seq) > 64:
            model.trunk.set_chunk_size(64)
            print("Optimization: Memory chunking enabled.")
        else:
            print("Optimization: Sequence small enough for direct processing.")
            
        inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False).to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)
            
            pdb_string = model.output_to_pdb(outputs)[0]
            # Calculate pLDDT correctly now that the float16 bug is bypassed
            plddt_score = outputs.plddt.mean().item() * 100 
            
        filename = "ww_domain_predicted.pdb"
        with open(filename, "w") as f:
            f.write(pdb_string)
            
        print("\n--- FOLDING COMPLETE ---")
        print(f"File saved: {filename}")
        print(f"Average Confidence (pLDDT): {plddt_score:.2f} / 100")
            
    except Exception as e:
        print(f"\nPipeline Failure: {e}")
        traceback.print_exc()
    finally:
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    fold_sequence()
