from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from vector_db_utils import load_settings, get_context, get_query_engine
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the root directory
ROOT_DIR = Path(__file__).parent

LLM_PATH = ROOT_DIR / "LLM"
LLM_TOKENIZER_PATH = ROOT_DIR / "LLM_TOKENIZER"

# Determine whether to load the CPU or GPU model
USE_GPU = torch.cuda.is_available()


class _SawserqGptService:
    """
    Singleton class for voice to text transformation using trained models
    """
    _instance = None

    tokenizer = None
    model = None
    context = None
    query_engine = None
    settings = load_settings()

    def query(self, query_str: str) -> str:
        # prompt (with context)
        prompt_template_w_context = lambda context, user_input: f"""[INST]SawserQGPT, functioning as a virtual Circassian history and culture expert, communicates in clear, accessible language, uses facts and reliable numbers upon request. \
        It reacts to feedback aptly and ends responses with its signature '–SawserQGPT'. \
        SawserQGPT will tailor its responses to match the user's input, providing concise acknowledgments to brief expressions of gratitude or feedback, \
        thus keeping the interaction natural and engaging.

        {context}
        Please respond to the following user's input. Use the context above if it is helpful.

        {user_input}
        [/INST]
        """
        print("running prompt:")
        print(f"query: {query_str}")
        print(f"settings: {str(self.settings)}")

        try:
            context = get_context(query_str, self.query_engine, self.settings["top_k"])
        except Exception as e:
            print(f"Failed to get context. Will use empty context. Error: {e}")
            context = ""

        prompt = prompt_template_w_context(context, query_str)

        if USE_GPU:
            device = "cuda"
        else:
            device = "cpu"

        print(f"device = {device}")

        print(f"model config = {str(self.model.config)}")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        print("done with tokenizer")

        outputs = self.model.generate(inputs=input_ids, max_new_tokens=280, device=device)

        return self.tokenizer.batch_decode(outputs)[0]

    @property
    def instance(self):
        return self._instance


def sawserq_gpt_service():
    """
    Factory function for _SawserqGptService class
    """
    if _SawserqGptService._instance is None:
        _SawserqGptService._instance = _SawserqGptService()

        print(f"USE_GPU is {USE_GPU}")

        print("Loading model...")

        model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

        # Load config from local path
        config = AutoConfig.from_pretrained(model_name)
        config.quantization_config["use_exllama"] = True
        config.quantization_config["exllama_config"] = {"version": 2}

        # disable exllama to be able to run on CPU
        # config = AutoConfig.from_pretrained(model_name)
        # if not USE_GPU:
        #     config.quantization_config["use_exllama"] = False

        _SawserqGptService.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                                        device_map="cuda",
                                                                        trust_remote_code=False,
                                                                        revision="main")
        # Load tokenizer
        _SawserqGptService.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        #
        # _SawserqGptService.tokenizer = AutoTokenizer.from_pretrained(str(LLM_TOKENIZER_PATH))
        # _SawserqGptService.model = AutoModelForCausalLM.from_pretrained(
        #     str(LLM_PATH),
        #     device_map="cuda:0",
        #     config=config
        # ).to("cuda" if USE_GPU else "cpu")
        print("Finished Loading model.")

        print("Loading Context Model...")
        _SawserqGptService.query_engine = get_query_engine(_SawserqGptService.settings)
        print("Finished Loading Context Model.")

    return _SawserqGptService._instance


if __name__ == "__main__":
    sgpt = sawserq_gpt_service()
    sqgpt_2 = sawserq_gpt_service()

    assert sgpt is sqgpt_2

    # make prediction
    answer = sgpt.query("Who are the Circassians?")
    print(answer)
