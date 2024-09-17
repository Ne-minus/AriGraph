from openai import OpenAI
from openai._types import NOT_GIVEN
from typing import List
from copy import deepcopy

from utils.contriever import Retriever
from prompts.assistant_prompts import (
    prompt_extraction_assistant,
    prompt_refining_items_assistant,
)
from utils.retriever_search_drafts import graph_retr_search
from utils.utils import (
    process_triplets,
    parse_triplets_removing,
    find_top_episodic_emb,
    top_k_obs,
    Logger,
)


def clear_triplet(triplet):
    return [
        triplet[0].lower().strip(""""'. `;:"""),
        triplet[1].lower().strip(""""'. `;:"""),
        {"label": triplet[2]["label"].lower().strip(""""'. `;:""")},
    ]


class AssistantGraph:
    def __init__(
        self, model: str, system_prompt: str, api_key: str, device: str = "cpu"
    ):
        self.triplets = []
        self.model = model
        self.system_prompt = system_prompt
        self.total_amount = 0
        self.client = OpenAI(api_key=api_key)
        self.retriever = Retriever(device)

    def generate(self, prompt: str, return_json: bool = False, t: float = 0.7):
        if return_json:
            response_format = {"type": "json_object"}
        else:
            response_format = NOT_GIVEN

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=self.model,
            response_format=response_format,
            temperature=t,
        )
        response = chat_completion.choices[0].message.content
        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

        cost = completion_tokens * 3 / 100000 + prompt_tokens * 1 / 100000
        cost = completion_tokens * 3 / 100000 + prompt_tokens * 1 / 100000
        self.total_amount += cost
        return response, cost

    @staticmethod
    def to_str(triplet: List):
        return triplet[0] + ", " + triplet[2]["label"] + ", " + triplet[1]

    def get_all_triplets(self):
        return [AssistantGraph.to_str(triplet) for triplet in self.triplets]

    def delete_all(self):
        self.triplets = []

    def exclude(self, triplets):
        new_triplets = []
        for triplet in triplets:
            triplet = clear_triplet(triplet)
            if triplet not in self.triplets:
                new_triplets.append(AssistantGraph.to_str(triplet))

        return new_triplets

    def delete_triplets(self, triplets):
        for triplet in triplets:
            if triplet in self.triplets:
                self.triplets.remove(triplet)

    def triplets_to_str(self, triplets):
        return [AssistantGraph.to_str(triplet) for triplet in triplets]

    def get_associated_triplets(self, items, steps=2):
        items = deepcopy([string.lower() for string in items])
        associated_triplets = []

        for i in range(steps):
            now = set()
            for triplet in self.triplets:
                for item in items:

                    if (item == triplet[0] or item == triplet[1]) and self.str(
                        triplet
                    ) not in associated_triplets:
                        associated_triplets.append(self.str(triplet))
                        if item == triplet[0]:
                            now.add(triplet[1])
                        if item == triplet[1]:
                            now.add(triplet[0])

                        break

            if "itself" in now:
                now.remove("itself")
            items = now
        return associated_triplets

    def get_embedding_local(self, text):
        return self.retriever.embed([text])[0].cpu().detach().numpy()

    def add_triplets(self, triplets):
        for triplet in triplets:
            triplet = clear_triplet(triplet)
            if triplet not in self.triplets:
                self.triplets.append(triplet)

    def update(
        self,
        user_input: str,
        prev_subgraph: set,
        new_entities_dict: dict,
        logger: Logger,
        topk_episodic: int = 2,
    ):
        # Extract triplets from user input with relevance scores
        prompt = prompt_extraction_assistant.format(
            user_input=user_input, prev_subgraph=prev_subgraph
        )
        response, _ = self.generate(prompt, t=0.001)

        # Preprocess extracted triplets
        new_triplets_raw = process_triplets(response)

        # Exclude new triplets that are already in the graph
        new_triplets = self.exclude(new_triplets_raw)

        logger("New triplets: " + str(new_triplets))
        new_entities = {triplet[0] for triplet in new_triplets_raw} | {
            triplet[1] for triplet in new_triplets_raw
        }

        # Get subgraph from new entities of the user input
        associated_subgraph = self.get_associated_triplets(new_entities, steps=1)

        # Update graph by removing outdated triplets
        prompt = prompt_refining_items_assistant.format(
            ex_triplets=associated_subgraph, new_triplets=new_triplets
        )
        response, _ = self.generate(prompt, t=0.001)
        predicted_outdated = parse_triplets_removing(response)
        self.delete_triplets(predicted_outdated)

        logger("Outdated triplets: " + response)
        logger("Number of replacements: " + str(len(predicted_outdated)))

        # Add new triplets to a graph
        self.add_triplets(new_triplets_raw)

        triplets = self.triplets_to_str(self.triplets)

        associated_subgraph = set()

        # Perform BFS for retrieving associated subgraph
        for query, depth in new_entities_dict.items():
            results = graph_retr_search(
                query,
                triplets,
                self.retriever,
                max_depth=depth,
                topk=6,
                post_retrieve_threshold=0.75,
                verbose=2,
            )
            associated_subgraph.update(results)

        # Return edges that are not in user's triplets already
        associated_subgraph = [
            element for element in associated_subgraph if element not in new_triplets
        ]

        return associated_subgraph
