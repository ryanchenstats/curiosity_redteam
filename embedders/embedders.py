import tempfile
import warnings
from pprint import pprint

import tiktoken
import torch

# import voyageai
# from InstructorEmbedding import INSTRUCTOR
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.settings import Settings
from openai import OpenAI
from ragatouille import RAGPretrainedModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


# from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# from https://huggingface.co/Salesforce/SFR-Embedding-Mistral
def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def chunk_text(texts: list[str], max_length=100_000):
    """A generate function that chunks a list of texts into sublists of total length <= max_length"""
    current_chunk = []
    chunk_len = 0

    for t in texts:
        if chunk_len + len(t) > max_length:
            yield current_chunk
            current_chunk = []
            chunk_len = 0
        current_chunk.append(t)
        chunk_len += len(t)

    # Yield the last sublist if it's not empty
    if current_chunk:
        yield current_chunk


class Embedder:
    def __call__(self, texts: list[str], **kwargs) -> torch.tensor:
        raise NotImplementedError

    def calculate_score(
        self, queries: str | list[str], texts: str | list[str]
    ) -> float:
        queries = [queries] if isinstance(queries, str) else queries
        texts = [texts] if isinstance(texts, str) else texts
        q_embed = self(queries)
        t_embed = self(texts)
        return cos_sim(q_embed, t_embed).mean().item()


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: SentenceTransformer = None,
        context_length: int | None = None,
    ):
        if model is not None:
            self.model = model
        else:
            print(model_name)
            self.model = SentenceTransformer(model_name)

        if context_length is not None:
            original_length = self.model.get_max_seq_length()
            assert context_length <= original_length
            self.model.max_seq_length = context_length
            assert self.model.get_max_seq_length() == context_length
            print(f"Set context_length to {context_length} from {original_length}")

    def __call__(self, texts: list[str] | str) -> torch.tensor:
        texts = [texts] if isinstance(texts, str) else texts
        return self.model.encode(texts, convert_to_tensor=True, batch_size=1024)


# class InstructorEmbedder(Embedder):
#     instruct = "Represent the Medical statement:"

#     def __init__(
#         self,
#         context_length: int | None = None,
#     ):
#         self.model = INSTRUCTOR("hkunlp/instructor-xl")

#         if context_length is not None:
#             original_length = self.model.get_max_seq_length()
#             assert context_length <= original_length
#             self.model.max_seq_length = context_length
#             assert self.model.get_max_seq_length() == context_length
#             print(f"Set context_length to {context_length} from {original_length}")

#     def __call__(self, texts: list[str] | str) -> torch.tensor:
#         texts = [texts] if isinstance(texts, str) else texts
#         texts = [[self.instruct, t] for t in texts]

#         return self.model.encode(texts, convert_to_tensor=True)


# class VoyageAIEmbedder(Embedder):
#     token_limit = {"voyage-2": 320_000, "voyage-large-2": 120_000}

#     def __init__(self, model_name: str = "voyage-2", context_length: int | None = None):
#         self.client = voyageai.Client()
#         self.model_name = model_name
#         self.context_length = context_length
#         if context_length is not None:
#             assert context_length <= self.token_limit[model_name]
#             print(
#                 f"Set context_length to {context_length} from {self.token_limit[model_name]}"
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 "voyageai/voyage", model_max_length=context_length
#             )

#     def __call__(self, texts: list[str] | str) -> torch.tensor:
#         texts = [texts] if isinstance(texts, str) else texts
#         if self.context_length is not None:
#             ids = self.tokenizer(texts, truncation=True)["input_ids"]
#             truncated_ids = [x[: self.context_length] for x in ids]
#             texts = self.tokenizer.batch_decode(
#                 truncated_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True,
#             )

#         assert len(texts) <= 128

#         # a crude estimate of text limit
#         text_limit = 0.8 * self.token_limit[self.model_name]
#         if (total_text_len := len("".join(texts))) > text_limit:
#             warnings.warn(
#                 f"Total input text length too long ({total_text_len}), splitting into chunks"
#             )
#             embeddings = []
#             for chunk in chunk_text(texts):
#                 embed = self.client.embed(chunk, model=self.model_name).embeddings
#                 embeddings.extend(embed)
#         else:
#             embeddings = self.client.embed(texts, model=self.model_name).embeddings
#         return torch.tensor(embeddings)


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        context_length: int | None = None,
    ):
        self.name = model_name
        self.client = OpenAI()
        self.context_length = context_length
        if context_length is not None:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
            print(f"Set context_length to {context_length}")

    def __call__(self, texts: list[str] | str) -> torch.tensor:
        texts = [texts] if isinstance(texts, str) else texts
        if self.context_length is not None:
            ids = self.tokenizer.encode_batch(texts)
            truncated_ids = [x[: self.context_length] for x in ids]
            texts = self.tokenizer.decode_batch(truncated_ids)
        response = self.client.embeddings.create(input=texts, model=self.name)
        embeddings = [x.embedding for x in response.data]
        return torch.tensor(embeddings)


class HuggingfaceEmbedder(Embedder):
    embed_pooling = ["cls", "mean"]
    default_context_length = {"medicalai/ClinicalBERT": 256}

    def __init__(
        self,
        model_name: str,
        embed_pooling: str,
        context_length: int | None = None,
    ):
        self.model = AutoModel.from_pretrained(model_name, device_map="cuda")
        assert embed_pooling in ["cls", "mean"]
        self.pool = embed_pooling
        default_context_length = self.default_context_length[model_name]
        if context_length is None:
            context_length = default_context_length
        else:
            assert context_length <= default_context_length
            print(
                f"Set context_length to {context_length} from {default_context_length}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=context_length
        )

    def __call__(self, texts: list[str] | str) -> torch.tensor:
        texts = [texts] if isinstance(texts, str) else texts
        input = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to("cuda")
        with torch.no_grad():
            output = self.model(**input)
        all_token_embed = output.last_hidden_state
        if self.pool == "mean":
            return mean_pooling(all_token_embed, input["attention_mask"])
        elif self.pool == "cls":
            return all_token_embed[:, 0]


class EmbeddingMistralEmbedder(Embedder):
    task_instruct: str = (
        "Given an expert response, retrieve relevant answers similar to the expert's response in medical context"
    )
    default_context_length: int = 4096

    def __init__(self, quantize: bool = False, context_length: int | None = None):
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        if context_length is None:
            context_length = self.default_context_length
        else:
            assert context_length <= self.default_context_length
            print(
                f"Set context_length to {context_length} from {self.default_context_length}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Salesforce/SFR-Embedding-Mistral",
            model_max_length=context_length,
        )
        self.model = AutoModel.from_pretrained(
            "Salesforce/SFR-Embedding-Mistral",
            device_map="cuda",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )

    @staticmethod
    def add_query_instruction(query: str, task_instruct: str | None = None) -> str:
        if task_instruct is None:
            task_instruct = EmbeddingMistralEmbedder.task_instruct
        return f"Instruct: {task_instruct}\nQuery: {query}"

    @staticmethod
    def add_batch_query_instruction(
        queries: list[str], task_instruct: str | None = None
    ) -> list[str]:
        return [
            EmbeddingMistralEmbedder.add_query_instruction(query, task_instruct)
            for query in queries
        ]

    def __call__(self, texts: list[str]) -> torch.tensor:
        texts = [texts] if isinstance(texts, str) else texts
        embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            start = i
            end = min(i + batch_size, len(texts))
            b_texts = texts[start:end]
            b_input = self.tokenizer(
                b_texts, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            with torch.no_grad():
                b_outputs = self.model(**b_input)
            b_embed = last_token_pool(
                b_outputs.last_hidden_state, b_input["attention_mask"]
            )
            embeddings.append(b_embed)
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def calculate_score(
        self, queries: str | list[str], texts: str | list[str]
    ) -> float:
        queries = [queries] if isinstance(queries, str) else queries
        texts = [texts] if isinstance(texts, str) else texts
        queries = self.add_batch_query_instruction(queries)
        q_embed = self(queries)
        t_embed = self(texts)
        return cos_sim(q_embed, t_embed).mean().item()


class LlamaIndexEmbedder(Embedder):
    def __init__(
        self,
        local_model: str | None = None,
        context_length: int | None = None,
    ):
        if local_model is not None:
            Settings.embed_model = resolve_embed_model(f"local:{local_model}")
        self.model = SemanticSimilarityEvaluator()
        if context_length is not None:
            raise NotImplementedError

    def calculate_score(
        self, queries: str | list[str], texts: str | list[str]
    ) -> float:
        queries = [queries] if isinstance(queries, str) else queries
        texts = [texts] if isinstance(texts, str) else texts
        response = texts
        reference = queries
        scores = []
        for resp in response:
            for ref in reference:
                result = self.model.evaluate(response=resp, reference=ref)
                scores.append(float(result.score))
        return sum(scores) / len(scores)


class ColBERT(Embedder):
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        context_length: int | None = None,
        agg: str = "max",
        search_method: str = "in_memory",
    ):
        self.rag = RAGPretrainedModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "colbert-ir/colbertv2.0", model_max_length=256
        )

        if agg == "mean":
            self.calculate_score = self.calcualte_mean_score
        elif agg == "max":
            self.calculate_score = self.calcualte_max_score
        else:
            raise ValueError

        if search_method == "in_memory":
            self.search = self.search_in_memory
        elif search_method == "building_index":
            self.search = self.search_with_building_index
        else:
            raise ValueError

        if context_length is not None:
            raise NotImplementedError

    def truncate(self, texts: list[str]) -> list[str]:
        ids = self.tokenizer(texts, truncation=True)["input_ids"]
        truncated_ids = [x[:256] for x in ids]
        texts = self.tokenizer.batch_decode(
            truncated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return texts

    def search_with_building_index(
        self, queries: list[str], documents: list[str], k: int
    ) -> list[list[dict]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.rag.index(
                collection=documents,
                index_name=tmpdir,
                split_documents=False,
            )
            out = self.rag.search(query=queries, k=k)
        return out

    def search_in_memory(
        self, queries: list[str], documents: list[str], k: int
    ) -> list[list[dict]]:
        return self.rag.rerank(query=queries, documents=documents, k=k, bsize=1024)

    def calcualte_mean_score(
        self, queries: str | list[str], texts: str | list[str]
    ) -> float:
        queries = [queries] if isinstance(queries, str) else queries
        texts = [texts] if isinstance(texts, str) else texts

        queries = self.truncate(queries)
        texts = self.truncate(texts)

        k = len(texts)
        out = self.search(queries, texts, k)
        n = 0
        total_score = 0
        for query in out:
            for x in query:
                total_score += x["score"]
                n += 1
        return total_score / n

    def calcualte_max_score(
        self, queries: str | list[str], texts: str | list[str]
    ) -> float:
        queries = [queries] if isinstance(queries, str) else queries
        texts = [texts] if isinstance(texts, str) else texts

        queries = self.truncate(queries)
        texts = self.truncate(texts)

        out = self.search(queries, texts, 1)
        scores = [x[0]["score"] for x in out]
        return sum(scores) / len(scores)


if __name__ == "__main__":
    embedder = ColBERT()
    t1 = ["hahaha"] * 3
    t2 = ["lalala"] * 1
    expert = [
        "Glasgow-Blatchford Score. Most recommend score 0-1 as threshold",
        "Glasgow Blatchord Score 0-1",
        "Blatchford Score",
    ]

    llm = [
        "Use the Glasgow-Blatchford score for risk stratification. Discharge patients with UGIB if their score is 0-1"
    ]
    perplexity = [
        "The Glasgow Blatchford Score (GBS) with a score of 0-1 performs best in identifying very low-risk patients with upper gastrointestinal bleeding (UGIB) who will not require hospital-based intervention or die, and is recommended by most guidelines to facilitate safe outpatient management.\n\nThe GBS is one of the most widely validated and recommended scores for pre-endoscopic risk stratification of UGIB. Several studies have shown that patients with a GBS ≤ 1 can be safely managed as outpatients without the need for endoscopic intervention, transfusion, or surgery, and have a very low risk of mortality.\n\nSpecifically, a GBS score of 0 has been associated with a very low risk of needing endoscopic therapy, radiologic intervention, surgery, rebleeding, or mortality at 30 days follow-up across multiple studies. A GBS ≤ 1 has also been used as the low-risk threshold for safe outpatient management in prospective studies.\n\nTherefore, the Glasgow Blatchford Score with a threshold of 0-1 is the recommended risk stratification tool to identify very low-risk UGIB patients who can potentially be discharged from the emergency department for outpatient management"
    ]

    print(embedder.calculate_score(expert, perplexity))