import argparse
from ctransformers import AutoModelForCausalLM as cAutoLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path

SAVED_MODEL_PATH = "../flask"
LLM_PATH = str(Path(SAVED_MODEL_PATH, "LLM"))
LLM_TOKENIZER_PATH = str(Path(SAVED_MODEL_PATH, "LLM_TOKENIZER"))
LLM_CPU_PATH = str(Path(SAVED_MODEL_PATH, "LLM_CPU"))


def load_llm_cpu():
    # model_name = "TheBloke/CodeLlama-7B-Instruct-GGUF"
    # model_file = "codellama-7b-instruct.Q6_K.gguf"
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    model_file = "mistral-7b-instruct-v0.2.Q6_K.gguf"
    llm = cAutoLLM.from_pretrained(model_name,
                                   model_file=model_file,
                                   model_type="llama",
                                   gpu_layers=0)
    return llm


def load_llm_gpu():
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
    parser = argparse.ArgumentParser(description='Load LLM model on CPU or GPU')
    parser.add_argument('mode', choices=['llm_cpu', 'llm_gpu'], help='Mode to load the LLM model')
    args = parser.parse_args()

    if args.mode == 'llm_cpu':
        llm = load_llm_cpu()
        Path(LLM_CPU_PATH).mkdir(parents=True, exist_ok=True)
        llm.save_pretrained(LLM_CPU_PATH)
    elif args.mode == 'llm_gpu':
        model, tokenizer = load_llm_gpu()
        Path(LLM_PATH).mkdir(parents=True, exist_ok=True)
        Path(LLM_TOKENIZER_PATH).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(LLM_PATH)
        tokenizer.save_pretrained(LLM_TOKENIZER_PATH)
