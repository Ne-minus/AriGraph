import os
from agents.parent_agent import GPTagent
from graphs.assistant_graph import AssistantGraph
from typing import List
from utils.utils import Logger


def get_prompt_from_subgraph(user_input: str, subgraph: List, history: List):
    return f"""\n1. History of {len(history)} last messages: {history} 
\n2. Your current message from the user: {user_input}
\n3. Information from the memory module that can be relevant to current dialogue: {subgraph}"""


def main():
    logger = Logger("assistant")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-4o-mini"
    agent = GPTagent(
        model=model,
        system_prompt="You are a helpful assistant",
        api_key=openai_api_key,
    )
    graph = AssistantGraph(
        model=model,
        system_prompt="You are a helpful assistant",
        api_key=openai_api_key,
    )
    prev_subgraph = []
    history = []
    while True:
        user_input = input()
        new_entities_dict, _ = agent.item_processing_scores_qa(user_input)
        new_entities_dict = {
            key.lower(): value for key, value in new_entities_dict.items()
        }
        logger(f"entities: {new_entities_dict}")
        subgraph = graph.update(
            user_input=user_input,
            prev_subgraph=prev_subgraph,
            new_entities_dict=new_entities_dict,
            logger=logger,
        )
        logger(f"subgraph: {subgraph}")
        prompt = get_prompt_from_subgraph(user_input, subgraph, history)
        response, _ = agent.generate(prompt, t=0.001)
        print(f"{response}")
        history.append(user_input)
        prev_subgraph = subgraph


if __name__ == "__main__":
    main()
