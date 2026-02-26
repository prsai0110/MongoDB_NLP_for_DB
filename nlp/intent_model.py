import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

class IntentModel:

    def __init__(self):
        self.labels = ["average","count","find","group","join","search","sum","summary"]

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("nlp/intent_model")
        self.model = DistilBertForSequenceClassification.from_pretrained("nlp/intent_model")

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

        return self.labels[predicted_class]
