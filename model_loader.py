from transformers import pipeline
import torch

def load_pipeline():
    """
    Loads the text-generation pipeline with the TinyLlama model.
    """
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model: {model_id}. This may take a moment...")

    try:
        # Create a text-generation pipeline
        # device_map="auto" will automatically use CPU if no GPU is available
        # torch_dtype is set to float32 for broad CPU compatibility
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

         # Ensure tokenizer has a pad token
        if pipe.tokenizer.pad_token is None:
            pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

        print("Model pipeline loaded successfully.")
        return pipe
        
    except Exception as e:
        print(f"Error loading model pipeline: {e}")
        print("Please ensure you have an internet connection for the first run,")
        print("and that you have installed the required libraries: pip install transformers torch accelerate")
        return None