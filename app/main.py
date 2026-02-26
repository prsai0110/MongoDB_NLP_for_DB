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
# 🔥 Response Formatter (NEW)
# -----------------------------------------------------

def format_response(intent, results):
    if not results:
        return "No records found."

    sentences = []

    if intent == "find":
        for item in results:
            name = item.get("name", "Unknown")
            dept = item.get("department", "Unknown department")
            salary = item.get("salary", "N/A")

            sentence = f"{name} works in {dept} department and earns ₹{salary}."
            sentences.append(sentence)

        return " ".join(sentences)

    elif intent == "aggregate":
        # For aggregation results
        if len(results) == 1:
            key, value = list(results[0].items())[0]
            return f"The {key} is {value}."

        return f"The query returned {len(results)} aggregated results."

    return "Query executed successfully."


# -----------------------------------------------------
# 📊 Chart Suggestion Engine (NEW)
# -----------------------------------------------------

def suggest_chart(intent, user_query):
    user_query = user_query.lower()

    if "count" in user_query or "how many" in user_query:
        return "pie"

    if "average" in user_query or "mean" in user_query:
        return "bar"

    if "trend" in user_query or "over time" in user_query:
        return "line"

    if "highest" in user_query or "top" in user_query:
        return "bar"

    if intent == "find":
        return "table"

    return "table"


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

        if any(word in user_query_lower for word in
               ["highest", "lowest", "descending", "ascending", "sorted"]):
            intent = "find"

        if (
            any(word in user_query_lower for word in ["above", "below", "greater than", "less than"])
            and not any(word in user_query_lower for word in
                        ["total", "sum", "average", "mean", "count"])
        ):
            intent = "find"

        if any(word in user_query_lower for word in
               ["total", "sum", "average", "mean", "count", "how many"]):
            intent = "aggregate"

        # -------------------------------------------------

        # 2️⃣ Schema matching
        collection, field = schema_matcher.match(user_query)

        # 3️⃣ Build MongoDB query
        mongo_query = query_builder.build(
            intent,
            collection,
            field,
            user_query
        )

        # 4️⃣ Execute query
        result = db.execute(mongo_query)

        if not isinstance(result, list):
            result = list(result)

        # -------------------------------------------------
        # 🔥 NEW LAYERS ADDED HERE
        # -------------------------------------------------

        english_summary = format_response(intent, result)
        chart_type = suggest_chart(intent, user_query)

        # -------------------------------------------------

        return {
            "status": "success",
            "user_query": user_query,
            "intent": intent,
            "collection": collection,
            "field": field,
            "mongo_query": mongo_query,
            "result_count": len(result),

            # ✅ NEW
            "summary": english_summary,
            "chart_type": chart_type,

            # ✅ Raw Data
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
