# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from detoxify import Detoxify

class Diversity():
    def __init__(self):
        # self.model = AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert')
        # self.tokenizer = AutoTokenizer.from_pretrained('unitary/toxic-bert')
        self.session = requests.Session()
        # self.API_URL = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
        # self.headers = {"Authorization": "Bearer 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"}
        self.model = Detoxify('original')
        self.classes = {
            'toxicity': 0,
            'severe_toxicity': 1,
            'obscene': 2,
            'threat': 3,
            'insult': 4,
            'identity_attack': 5
        }
        self.total_classes = len(self.classes)
        
    # def __call__(self, x: list[str]) -> list[torch.Tensor]:
    #     inputs = self.tokenizer(x, truncation=True, padding=True, return_tensors='pt')
    #     with torch.no_grad():
    #         output = self.model(**inputs).logits
    #     probs = torch.nn.functional.softmax(output)
    #     return probs
    
    def __call__(self, x: list[str]) -> list[torch.Tensor]:
        if isinstance(x, str):
            x = [x]
        output = self.model.predict(x)
        # output = output.json()
        num_out = len(x)
        scores = torch.zeros((num_out, self.total_classes + 1))
        for tclass, score in output.items():
            scores[:, self.classes[tclass]] = torch.Tensor(score)
        # for k, out in enumerate(output):
        #     for class_score in out:
        #         try:
        #             toxic_class = class_score['label']
        #         except:
        #             print(x)
        #             print(out)
        #             raise KeyError
        #         score = class_score['score']    
        #         scores[k, self.classes[toxic_class]] = score
        #     scores[k,-1] = 1 - scores[k,:-1].sum(dim=-1)
        scores = scores.clamp(0,1)
        scores[scores == 0] = 1e-6
        entropy = -torch.sum(scores * torch.log(scores), dim=-1)
        return entropy

if __name__ == '__main__':
    diversity = Diversity()
    output = diversity(["I love you", "I hate you"])
    print(output)