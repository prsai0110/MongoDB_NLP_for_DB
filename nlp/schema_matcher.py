import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SchemaMatcher:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        self.schema = {
            "sales": ["amount", "year", "category", "customer_id"],
            "customers": ["name", "city", "age"]
        }

        self.embeddings = self._embed_schema()

    def _embed_schema(self):
        embedded = {}
        for collection, fields in self.schema.items():
            embedded[collection] = {
                field: self.embedding_model.encode(field)
                for field in fields
            }
        return embedded

    def match(self, text):

        query_vector = self.embedding_model.encode(text)

        best_match = None
        highest_score = -1

        for collection, fields in self.embeddings.items():
            for field, vector in fields.items():

                score = cosine_similarity(
                    [query_vector], [vector]
                )[0][0]

                if score > highest_score:
                    highest_score = score
                    best_match = (collection, field)

        return best_match
