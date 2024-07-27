from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
    settings = load_settings()

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

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

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
        print("Loading Tokenizer ...")
        _SawserqGptService.tokenizer = AutoTokenizer.from_pretrained(str(LLM_TOKENIZER_PATH))
        print("Finished Loading Tokenizer.")
        print("Loading LLM ...")
        # disable exllama to be able to run on CPU
        config = AutoConfig.from_pretrained(str(LLM_PATH))
        config.quantization_config["use_exllama"] = False
        _SawserqGptService.model = AutoModelForCausalLM.from_pretrained(str(LLM_PATH), config=config)
        print("Finished Loading LLM.")
        print("Loading Context Model...")
        _SawserqGptService.context = lambda user_input: get_context(query=user_input,
                                                                    query_engine=get_query_engine(_SawserqGptService.settings),
                                                                    top_k=get_query_engine(["top_k"]))
        print("Finished Loading Context Model...")

    return _SawserqGptService._instance


if __name__ == "__main__":
    sgpt = sawserq_gpt_service()
    sqgpt_2 = sawserq_gpt_service()

    assert sgpt is sqgpt_2

    # make prediction
    answer = sgpt.query("Who are the Circassians?")
    print(answer)
