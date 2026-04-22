def build_search_tool() -> dict:
    return {
        "name": "search_database",
        "description": "Search PostgreSQL database for government service capabilities matching an RFQ task requirement",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Query string describing the task or capability to search for, e.g. 'cloud infrastructure support'",
                },
                "classification_filter": {
                    "type": "object",
                    "description": "Filter results by top-level classification, e.g. 'IT Services' or 'Professional Services'",
                    "properties": {
                        "comparison_operator": {
                            "type": "string",
                            "description": "Operator to compare the column value, either '=' or '!='",
                        },
                        "value": {
                            "type": "string",
                            "description": "Classification name, e.g. 'IT Services'",
                        },
                    },
                },
                "category_filter": {
                    "type": "object",
                    "description": "Filter results by capability category, e.g. 'Engineering' or 'Analytics'",
                    "properties": {
                        "comparison_operator": {
                            "type": "string",
                            "description": "Operator to compare the column value, either '=' or '!='",
                        },
                        "value": {
                            "type": "string",
                            "description": "Category name, e.g. 'Engineering'",
                        },
                    },
                },
            },
            "required": ["search_query"],
        },
    }
