from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

from models import Request

class DataProcessor:
    def __init__(self, tokenizer, dataset_name: str, max_rows: int | None = 100):
        self.tokenizer = tokenizer

        dataset = load_dataset(dataset_name, split="train", streaming=True)
        if max_rows is not None:
            dataset = dataset.take(max_rows)
        self.data_df = pd.DataFrame(dataset)
    
    def process(self, row_id: int):
        conversation = self.data_df['conversations'][row_id]
        requests = []
        l_kv = 0
        for i in range(int(len(conversation) / 2)):
            l_qo = len(self.tokenizer.encode(conversation[i]['value']))
            resp_len = len(self.tokenizer.encode(conversation[i + 1]['value']))
            
            l_kv += l_qo
            requests.append(
                Request(
                    id=i,
                    prompt_len=l_qo,
                    response_len=resp_len,
                    l_qo=l_qo,
                    l_kv=l_kv
                )
            )
            l_kv += resp_len

        return requests

    def process_first_request_each_row(self, num_rows: int | None = None) -> list[Request]:
        requests = []
        rows = len(self.data_df) if num_rows is None else min(num_rows, len(self.data_df))

        for row_id in range(rows):
            conversation = self.data_df["conversations"][row_id]
            if len(conversation) < 2:
                continue

            l_qo = len(self.tokenizer.encode(conversation[0]["value"]))
            resp_len = len(self.tokenizer.encode(conversation[1]["value"]))
            requests.append(
                Request(
                    id=row_id,
                    prompt_len=l_qo,
                    response_len=resp_len,
                    l_qo=l_qo,
                    l_kv=l_qo,
                )
            )

        return requests

    def process_request_from_each_row(self, num_rows: int | None = None) -> list[Request]:
        requests = []
        rows = len(self.data_df) if num_rows is None else min(num_rows, len(self.data_df))

        for row_id in range(rows):
            conversation = self.data_df["conversations"][row_id]
            if len(conversation) < 2:
                continue

            l_kv = 0
            for turn_id in range(0, len(conversation) - 1, 2):
                l_qo = len(self.tokenizer.encode(conversation[turn_id]["value"]))
                resp_len = len(self.tokenizer.encode(conversation[turn_id + 1]["value"]))
                l_kv += l_qo

                if turn_id == 0:
                    requests.append(
                        Request(
                            id=row_id,
                            prompt_len=l_qo,
                            response_len=resp_len,
                            l_qo=l_qo,
                            l_kv=l_kv,
                        )
                    )
                    break

                l_kv += resp_len

        return requests


def load_requests(
    row_id: int = 0,
    dataset_name: str = "Alignment-Lab-AI/CodeInterpreterData-sharegpt",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    max_rows: int | None = 100,
) -> list[Request]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dp = DataProcessor(tokenizer, dataset_name, max_rows=max_rows)
    return dp.process(row_id)


def load_batch_requests(
    num_rows: int = 100,
    dataset_name: str = "Alignment-Lab-AI/CodeInterpreterData-sharegpt",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
) -> list[Request]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dp = DataProcessor(tokenizer, dataset_name, max_rows=num_rows)
    return dp.process_first_request_each_row(num_rows=num_rows)


def load_row_requests(
    num_rows: int = 100,
    dataset_name: str = "Alignment-Lab-AI/CodeInterpreterData-sharegpt",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
) -> list[Request]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dp = DataProcessor(tokenizer, dataset_name, max_rows=num_rows)
    return dp.process_request_from_each_row(num_rows=num_rows)

if __name__ == "__main__":
    out = load_requests(0)
    for o in out:
        print(f"id: {o.id}; prompt_len: {o.prompt_len}; response_len: {o.response_len}; l_qo: {o.l_qo}; l_kv: {o.l_kv}")
