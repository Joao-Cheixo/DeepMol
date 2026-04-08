import numpy as np
from rdkit.Chem import MolToSmiles, MolFromSmiles
import logging
import os

# using safetensors
os.environ["TRANSFORMERS_USE_SAFETENSORS"] = "true"

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _transformers_available = True
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    _transformers_available = False
    logger.warning("Transformers not available.")


class HuggingFaceFeaturizer:
    """
    HuggingFaceFeaturizer that works with torch 2.5.1 using SafeTensors
    """
    
    def __init__(self, 
                 model_name: str = "seyonec/ChemBERTa-zinc-base-v1",  # Use this model instead
                 pooling: str = "mean",
                 max_length: int = 512):
        
        if not _transformers_available:
            raise ImportError("Transformers and torch are required for HuggingFaceFeaturizer")
        
        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length
        
        print(f"Loading model: {model_name}")
        print(" Using SafeTensors format to avoid torch.load issues...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with explicit safetensors preference
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True  # Explicitly force safetensors
            )
            self.model.eval()
            
            # Initialize feature names
            embedding_dim = self.model.config.hidden_size
            self.features_names = [f'chemberta_{i}' for i in range(embedding_dim)]
            
            print(f" Successfully loaded ChemBERTa model: {model_name}")
            print(f" Embedding dimension: {embedding_dim}")
            
        except Exception as e:
            print(f" Failed to load model: {e}")
            print(" Trying alternative model...")
            self._try_alternative_model()
    
    def _try_alternative_model(self):
        """Try loading an alternative model that's known to work"""
        alternative_models = [
            "seyonec/ChemBERTa-zinc-base-v1",
            "seyonec/PubChem10M_SMILES_BPE_396_250",
            "laituan245/molt5-base"
        ]
        
        for model_name in alternative_models:
            try:
                print(f" Trying alternative model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                self.model.eval()
                
                embedding_dim = self.model.config.hidden_size
                self.features_names = [f'chemberta_{i}' for i in range(embedding_dim)]
                self.model_name = model_name
                
                print(f" Successfully loaded alternative model: {model_name}")
                return
                
            except Exception as e:
                print(f" Failed with {model_name}: {e}")
                continue
        
        raise ImportError("Could not load any ChemBERTa model with current torch version")
    
    def featurize_smiles(self, smiles: str):
        """Featurize a single SMILES string."""
        try:
            if not smiles or not isinstance(smiles, str):
                return None
                
            # Tokenize
            inputs = self.tokenizer(
                smiles,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
            
            # Apply pooling
            if self.pooling == "mean":
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            elif self.pooling == "cls":
                embedding = token_embeddings[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling: {self.pooling}")
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.warning(f"Failed to featurize SMILES {smiles}: {str(e)}")
            return None


# Test the featurizer
if __name__ == "__main__":
    print(" Testing HuggingFaceFeaturizer with Torch 2.5.1...")
    print("=" * 50)
    
    # Test molecules
    test_smiles = [
        "CCO",  # Ethanol
        "C1=CC=CC=C1",  # Benzene
        "CC(=O)O",  # Acetic acid
    ]
    
    try:
        featurizer = HuggingFaceFeaturizer()
        
        print(f"\n Testing {len(test_smiles)} molecules:")
        print("=" * 50)
        
        for i, smiles in enumerate(test_smiles, 1):
            print(f"\n{i}. Processing: {smiles}")
            embedding = featurizer.featurize_smiles(smiles)
            if embedding is not None:
                print(f"   Embedding shape: {embedding.shape}")
                print(f"   First 5 values: {[f'{x:.4f}' for x in embedding[:5]]}")
            else:
                print("    Failed to generate embedding")
                
    except Exception as e:
        print(f" Error: {e}")
        print(" Alternative solutions:")
        print("   1. Try upgrading only torch: pip install torch==2.6.1 --upgrade")
        print("   2. Clear HuggingFace cache: huggingface-cli delete-cache")
        print("   3. Use a local model path if available")