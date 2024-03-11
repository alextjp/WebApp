# llm_chain.py
# Basic structure for LLM Chain

class LLMChain:
    def __init__(self, llm, prompt, model_name, api_key):
        self.llm = llm
        self.prompt = prompt
        self.model_name = model_name
        self.api_key = api_key