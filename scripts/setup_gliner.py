#!/usr/bin/env python
"""
Download and test GLiNER models for the routing system
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def download_gliner_models():
    """Download all GLiNER models used in the routing system"""
    try:
        from gliner import GLiNER
        import torch
        
        models = [
            "urchade/gliner_multi-v2.1",
            "urchade/gliner_large-v2.1", 
            "urchade/gliner_medium-v2.1",
            "urchade/gliner_small-v2.1"
        ]
        
        print("🚀 Downloading GLiNER models...")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        for model_name in models:
            print(f"\n📥 Downloading {model_name}...")
            try:
                model = GLiNER.from_pretrained(model_name)
                
                # Test the model
                test_text = "Show me videos about machine learning"
                test_labels = ["video_content", "text_content", "machine_learning"]
                
                entities = model.predict_entities(test_text, test_labels, threshold=0.3)
                print(f"✅ {model_name} loaded successfully")
                print(f"   Test result: {entities}")
                
                del model  # Free memory
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        print("\n✨ GLiNER setup complete!")
        return True
        
    except ImportError:
        print("❌ GLiNER not installed. Installing...")
        os.system("pip install gliner")
        return download_gliner_models()
    except Exception as e:
        print(f"❌ Error setting up GLiNER: {e}")
        return False

if __name__ == "__main__":
    success = download_gliner_models()
    sys.exit(0 if success else 1)