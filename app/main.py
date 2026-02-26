from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nlp.intent_model import IntentModel
from nlp.embedding_model import EmbeddingModel
from nlp.schema_matcher import SchemaMatcher
from query_engine.query_builder import QueryBuilder
from db.connection import MongoConnection


# -----------------------------------------------------
# FastAPI App
# -----------------------------------------------------

app = FastAPI(title="MongoDB NLP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Initialize Components (Load Once)
# -----------------------------------------------------

intent_model = IntentModel()
embedding_model = EmbeddingModel()
schema_matcher = SchemaMatcher(embedding_model)
query_builder = QueryBuilder()
db = MongoConnection()


# -----------------------------------------------------
# Root Endpoint
# -----------------------------------------------------

@app.get("/")
def root():
    return {"message": "MongoDB NLP API is running successfully 🚀"}


# -----------------------------------------------------
# Query Endpoint
# -----------------------------------------------------

@app.get("/query")
def process_query(user_query: str):

    try:
        # 1️⃣ Detect intent using ML model
        intent = intent_model.predict(user_query)

        # -------------------------------------------------
        # 🔥 Smart Intent Correction Layer (Hybrid AI)
        # -------------------------------------------------

        user_query_lower = user_query.lower()

        # Sorting queries should be FIND
        if any(word in user_query_lower for word in
               ["highest", "lowest", "descending", "ascending", "sorted"]):
            intent = "find"

        # Filter-only queries without aggregation keywords
        if (
            any(word in user_query_lower for word in ["above", "below", "greater than", "less than"])
            and not any(word in user_query_lower for word in
                        ["total", "sum", "average", "mean", "count"])
        ):
            intent = "find"

        # Explicit aggregation words override everything
        if any(word in user_query_lower for word in
               ["total", "sum", "average", "mean", "count", "how many"]):
            intent = "aggregate"

        # -------------------------------------------------

        # 2️⃣ Schema matching (collection + field)
        collection, field = schema_matcher.match(user_query)

        # 3️⃣ Build dynamic MongoDB query
        mongo_query = query_builder.build(
            intent,
            collection,
            field,
            user_query
        )

        # 4️⃣ Execute query
        result = db.execute(mongo_query)

        # Convert cursor to list if needed
        if not isinstance(result, list):
            result = list(result)

        return {
            "status": "success",
            "user_query": user_query,
            "intent": intent,
            "collection": collection,
            "field": field,
            "mongo_query": mongo_query,
            "result_count": len(result),
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )