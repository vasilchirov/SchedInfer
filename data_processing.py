from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

from models import Request

class DataProcessor:
    def __init__(self, tokenizer, dataset_name: str):
        self.tokenizer = tokenizer

        dataset = load_dataset(dataset_name, split="train", streaming=True).take(100)
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

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1"
    )

    dp = DataProcessor(tokenizer, "Alignment-Lab-AI/CodeInterpreterData-sharegpt")
    out = dp.process(0)
    for o in out:
        print(f"id: {o.id}; prompt_len: {o.prompt_len}; response_len: {o.response_len}; l_qo: {o.l_qo}; l_kv: {o.l_kv}")
