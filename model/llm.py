from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path

SAVED_MODEL_PATH = "../flask"
LLM_PATH = str(Path(SAVED_MODEL_PATH, "LLM"))
LLM_TOKENIZER_PATH = str(Path(SAVED_MODEL_PATH, "LLM_TOKENIZER"))


def load_llm():
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

    # disable exllama to be able to run on CPU
    config = AutoConfig.from_pretrained(model_name)
    config.quantization_config["use_exllama"] = False

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Ensure this is configured for GPU usage if available
        trust_remote_code=False,
        revision="main",
        config=config  # Pass the config
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer


if __name__ == "__main__":
    # Ensure that directories exist
    Path(LLM_PATH).mkdir(parents=True, exist_ok=True)
    Path(LLM_TOKENIZER_PATH).mkdir(parents=True, exist_ok=True)

    # Save models locally
    model, tokenizer = load_llm()
    model.save_pretrained(LLM_PATH)
    tokenizer.save_pretrained(LLM_TOKENIZER_PATH)