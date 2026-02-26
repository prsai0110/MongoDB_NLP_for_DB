import re


class QueryBuilder:

    def build(self, intent, collection, field, user_query=None):

        user_query = user_query.lower() if user_query else ""
        pipeline = []
        filter_stage = {}
        sort_stage = None
        group_stage = None
        facet_stage = None

        # -------------------------------------------------
        # STEP 1: DETECT NUMERIC FILTERS (above / below / equal)
        # -------------------------------------------------

        numbers = re.findall(r"\d+", user_query)
        number = int(numbers[0]) if numbers else None

        if number is not None:

            if "above" in user_query or "greater than" in user_query:
                filter_stage[field] = {"$gt": number}

            elif "below" in user_query or "less than" in user_query:
                filter_stage[field] = {"$lt": number}

            elif intent == "find":
                filter_stage[field] = number

        # -------------------------------------------------
        # STEP 2: SORTING
        # -------------------------------------------------

        if "highest" in user_query or "descending" in user_query:
            sort_stage = {field: -1}

        elif "lowest" in user_query or "ascending" in user_query:
            sort_stage = {field: 1}

        # -------------------------------------------------
        # STEP 3: AGGREGATION DETECTION
        # -------------------------------------------------

        if "total" in user_query or "sum" in user_query:
            group_stage = {
                "_id": None,
                "total": {"$sum": f"${field}"}
            }

        elif "average" in user_query or "mean" in user_query:
            group_stage = {
                "_id": None,
                "average": {"$avg": f"${field}"}
            }

        elif "count" in user_query or "how many" in user_query or "number of" in user_query:
            group_stage = {
                "_id": None,
                "count": {"$sum": 1}
            }

        # -------------------------------------------------
        # STEP 4: GROUP BY SUPPORT
        # Example: group sales by customer above 1000
        # -------------------------------------------------

        if "group by" in user_query or "per" in user_query:

            group_field_match = re.search(r"group by (\w+)", user_query)

            if group_field_match:
                group_field = group_field_match.group(1)
            else:
                group_field = field

            group_stage = {
                "_id": f"${group_field}",
                "total": {"$sum": f"${field}"}
            }

        # -------------------------------------------------
        # STEP 5: BUILD PIPELINE ORDER
        # -------------------------------------------------

        if filter_stage:
            pipeline.append({"$match": filter_stage})

        if group_stage:
            pipeline.append({"$group": group_stage})

        if sort_stage:
            pipeline.append({"$sort": sort_stage})

        # -------------------------------------------------
        # STEP 6: FACET SUPPORT
        # Example:
        # "show sales above 2000 with total and highest"
        # -------------------------------------------------

        if "facet" in user_query or ("total" in user_query and "highest" in user_query):

            facet_stage = {
                "$facet": {
                    "totalData": [
                        {"$group": {
                            "_id": None,
                            "total": {"$sum": f"${field}"}
                        }}
                    ],
                    "highestData": [
                        {"$sort": {field: -1}},
                        {"$limit": 1}
                    ]
                }
            }

            return {
                "aggregate": collection,
                "pipeline": [facet_stage]
            }

        # -------------------------------------------------
        # STEP 7: JOIN SUPPORT (UNCHANGED)
        # -------------------------------------------------

        if intent == "join":
            return {
                "aggregate": collection,
                "pipeline": [
                    {
                        "$lookup": {
                            "from": "customers",
                            "localField": "customer_id",
                            "foreignField": "_id",
                            "as": "customer_info"
                        }
                    }
                ]
            }

        # -------------------------------------------------
        # STEP 8: RETURN LOGIC
        # -------------------------------------------------

        # If we built aggregation stages
        if pipeline:
            return {
                "aggregate": collection,
                "pipeline": pipeline
            }

        # Basic find fallback
        if intent == "find":
            return {
                "find": collection,
                "filter": {}
            }

        return {
            "find": collection,
            "filter": {}
        }