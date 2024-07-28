import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ctransformers import AutoModelForCausalLM as cAutoLLM
from pathlib import Path
from vector_db_utils import load_settings, get_context, get_query_engine

# Define the root directory
ROOT_DIR = Path(__file__).parent

LLM_PATH = ROOT_DIR / "LLM"
LLM_TOKENIZER_PATH = ROOT_DIR / "LLM_TOKENIZER"
LLM_CPU_PATH = ROOT_DIR / "LLM_CPU"


class _SawserqGptService:
    """
    Singleton class for voice to text transformation using trained models
    """
    _instance = None

    tokenizer = None
    model = None
    context = None
    settings = load_settings()
    use_gpu = None

    def query(self, query: str) -> str:
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

        context = self.context(query)
        prompt = prompt_template_w_context(context, query)

        if self.use_gpu:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
            return self.tokenizer.batch_decode(outputs)[0]
        else:
            output = self.model(prompt)
            return output

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
        _SawserqGptService.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"

        if _SawserqGptService.use_gpu:
            print("Loading GPU model...")
            model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
            config = AutoConfig.from_pretrained(model_name)
            config.quantization_config["use_exllama"] = False
            _SawserqGptService.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            _SawserqGptService.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
                config=config
            )
            print("Finished Loading GPU model.")
        else:
            print("Loading CPU model...")
            _SawserqGptService.model = cAutoLLM.from_pretrained(str(LLM_CPU_PATH))
            print("Finished Loading CPU model.")

        print("Loading Context Model...")
        _SawserqGptService.context = lambda user_input: get_context(
            query=user_input,
            query_engine=get_query_engine(_SawserqGptService.settings),
            top_k=get_query_engine(["top_k"])
        )
        print("Finished Loading Context Model.")

    return _SawserqGptService._instance


if __name__ == "__main__":
    sgpt = sawserq_gpt_service()
    sqgpt_2 = sawserq_gpt_service()

    assert sgpt is sqgpt_2

    # make prediction
    answer = sgpt.query("Who are the Circassians?")
    print(answer)
