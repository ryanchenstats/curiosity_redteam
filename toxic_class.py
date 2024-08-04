# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests


class Diversity():
    def __init__(self):
        # self.model = AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert')
        # self.tokenizer = AutoTokenizer.from_pretrained('unitary/toxic-bert')
        self.session = requests.Session()
        self.API_URL = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
        self.headers = {"Authorization": "Bearer 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"}
        
    # def __call__(self, x: list[str]) -> list[torch.Tensor]:
    #     inputs = self.tokenizer(x, truncation=True, padding=True, return_tensors='pt')
    #     with torch.no_grad():
    #         output = self.model(**inputs).logits
    #     probs = torch.nn.functional.softmax(output)
    #     return probs
    
    def __call__(self, x: list[str]) -> list[torch.Tensor]:
        output = self.session.post(self.API_URL,
                                   headers=self.headers,
                                   json={'inputs': x})
        return output.json()

if __name__ == '__main__':
    diversity = Diversity()
    print(diversity(["I love you", "I hate you"]))  
    