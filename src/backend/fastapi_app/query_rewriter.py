import json

from openai.types.responses import Response, ResponseFunctionToolCall


def build_search_function() -> dict:
    return {
        "type": "function",
        "name": "search_database",
        "description": "Search PostgreSQL database for government service capabilities matching an RFQ task requirement",
        "parameters": {
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


def extract_search_arguments(original_user_query: str, response: Response):
    search_query = None
    filters = []
    tool_calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
    if tool_calls:
        for tool_call in tool_calls:
            if tool_call.name == "search_database":
                arg = json.loads(tool_call.arguments)
                search_query = arg.get("search_query", original_user_query)
                if "classification_filter" in arg and arg["classification_filter"] and isinstance(arg["classification_filter"], dict):
                    cf = arg["classification_filter"]
                    filters.append(
                        {
                            "column": "classification_name",
                            "comparison_operator": cf["comparison_operator"],
                            "value": cf["value"],
                        }
                    )
                if "category_filter" in arg and arg["category_filter"] and isinstance(arg["category_filter"], dict):
                    catf = arg["category_filter"]
                    filters.append(
                        {
                            "column": "category_name",
                            "comparison_operator": catf["comparison_operator"],
                            "value": catf["value"],
                        }
                    )
    elif response.output_text:
        search_query = response.output_text.strip()
    return search_query, filters
