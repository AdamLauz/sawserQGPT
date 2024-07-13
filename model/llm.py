from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

SAVED_MODEL_PATH = "../flask"
LLM_PATH = str(Path(SAVED_MODEL_PATH, "LLM.h5"))
LLM_TOKENIZER_PATH = str(Path(SAVED_MODEL_PATH, "LLM_TOKENIZER.h5"))


def load_llm():
    # import LLM
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")

    config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
    model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer


if __name__ == "__main__":
    # save models locally
    model, tokenizer = load_llm()
    model.save_pretrained(LLM_PATH)
    tokenizer.save_pretrained(LLM_TOKENIZER_PATH)
