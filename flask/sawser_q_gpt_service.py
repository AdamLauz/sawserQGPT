import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ctransformers import AutoModelForCausalLM as cAutoLLM
from pathlib import Path
from vector_db_utils import load_settings, get_context, get_query_engine

# Define the root directory
ROOT_DIR = Path(__file__).parent

LLM_PATH = ROOT_DIR / "LLM"
LLM_TOKENIZER_PATH = ROOT_DIR / "LLM_TOKENIZER"


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
    use_gpu = None

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
        context = get_context(query_str, self.query_engine, self.settings["top_k"])
        prompt = prompt_template_w_context(context, query_str)

        if self.use_gpu:
            device = "cuda"
        else:
            device = "cpu"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=280)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def instance(self):
        return self._instance


def sawserq_gpt_service():
    """
    Factory function for _SawserqGptService class
    """
    if _SawserqGptService._instance is None:
        _SawserqGptService._instance = _SawserqGptService()

        # Determine whether to load the CPU or GPU model
        _SawserqGptService.use_gpu = os.getenv("USE_GPU", "false").lower()

        print(f"USE_GPU is {_SawserqGptService.use_gpu }")

        print("Loading model...")
        model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        # Load config from local path
        config = AutoConfig.from_pretrained(str(LLM_PATH))

        _SawserqGptService.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _SawserqGptService.model = AutoModelForCausalLM.from_pretrained(
            str(LLM_PATH),
            config=config
        )
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
