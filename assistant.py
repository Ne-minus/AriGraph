import os
from agents.parent_agent import GPTagent
from graphs.assistant_graph import AssistantGraph
from typing import List
from utils.utils import Logger, get_qa_prompt_from_subgraph


class Assistant:
    def __init__(self, model: str, api_key: str):
        self.agent = GPTagent(
            model=model,
            system_prompt="You are a helpful assistant",
            api_key=api_key,
        )
        self.graph = AssistantGraph(
            model=model,
            system_prompt="You are a helpful assistant",
            api_key=api_key,
        )
        self.prev_subgraph = []
        self.history = []

    def update_from_str(self, text: str, logger: Logger):
        self.graph.update(text, logger)

    def update_from_file(self, filepath: str, logger: Logger):
        with open(filepath, "r") as f:
            text = f.read()
        self.update_from_str(text, logger)

    def answer(self, user_input: str, logger: Logger):
        new_entities_dict, _ = self.agent.item_processing_scores_qa(user_input)
        new_entities_dict = {
            key.lower(): value for key, value in new_entities_dict.items()
        }
        logger(f"entities: {new_entities_dict}")
        subgraph, top_episodic = self.graph.update_and_retrieve(
            user_input=user_input,
            prev_subgraph=self.prev_subgraph,
            new_entities_dict=new_entities_dict,
            logger=logger,
        )
        logger(f"subgraph: {subgraph}")
        prompt = get_qa_prompt_from_subgraph(
            user_input, subgraph, top_episodic, self.history
        )
        response, _ = self.agent.generate(prompt, t=1.0)
        self.history.append(user_input)
        self.prev_subgraph = subgraph
        return response


logger = Logger("assistant")
openai_api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4o-mini"

assistant = Assistant(model, openai_api_key)
graph_filepath = "../graph.pickle"
# assistant.update_from_file("documents/yandex_internship.txt", logger)
# assistant.graph.save_to_file(graph_filepath)
assistant.graph.load_from_file(graph_filepath)


def chat_response(user_input):
    return assistant.answer(user_input, logger)
