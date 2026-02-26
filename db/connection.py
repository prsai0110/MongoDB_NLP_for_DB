from pymongo import MongoClient
import certifi


class MongoConnection:

    def __init__(self):

        self.client = MongoClient(
            "mongodb+srv://nlp_user:1234@cluster0.nizgx7n.mongodb.net/?retryWrites=true&w=majority",
            tls=True,
            tlsCAFile=certifi.where()
        )

        self.db = self.client["nlp_database"]

    def execute(self, query):

        # -------------------------
        # FIND (with filter + sort)
        # -------------------------
        if "find" in query:

            collection = query["find"]
            filter_query = query.get("filter", {})
            sort_query = query.get("sort", None)

            cursor = self.db[collection].find(filter_query, {"_id": 0})

            if sort_query:
                field, direction = list(sort_query.items())[0]
                cursor = cursor.sort(field, direction)

            return list(cursor)

        # -------------------------
        # COUNT (with optional filter)
        # -------------------------
        if "count" in query:

            collection = query["count"]
            filter_query = query.get("filter", {})

            return {
                "count": self.db[collection].count_documents(filter_query)
            }

        # -------------------------
        # AGGREGATE (group, sum, avg, join, etc.)
        # -------------------------
        if "aggregate" in query:

            return list(
                self.db[query["aggregate"]].aggregate(query["pipeline"])
            )

        return {"message": "Unsupported query"}