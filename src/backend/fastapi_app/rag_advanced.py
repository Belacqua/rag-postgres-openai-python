from collections.abc import AsyncGenerator
from typing import Optional

import anthropic

from fastapi_app.api_models import (
    CapabilityPublic,
    CategoryFilter,
    ChatRequestOverrides,
    ClassificationFilter,
    Filter,
    RAGContext,
    RetrievalResponse,
    RetrievalResponseDelta,
    ThoughtStep,
)
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.query_rewriter import build_search_tool
from fastapi_app.rag_base import RAGChatBase


class AdvancedRAGChat(RAGChatBase):
    query_prompt_template = open(RAGChatBase.prompts_dir / "query.txt").read()

    def __init__(
        self,
        *,
        messages: list[dict],
        overrides: ChatRequestOverrides,
        searcher: PostgresSearcher,
        anthropic_chat_client: anthropic.AsyncAnthropic,
        chat_model: str,
        chat_deployment: Optional[str] = None,
    ):
        self.searcher = searcher
        self.anthropic_chat_client = anthropic_chat_client
        self.chat_model = chat_model
        self.chat_params = self.get_chat_params(messages, overrides)
        self.model_for_thoughts = {"model": chat_model}

    def _extract_search_arguments(self, original_query: str, response) -> tuple[str, list[Filter]]:
        search_query = original_query
        filters: list[Filter] = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "search_database":
                inp = block.input
                search_query = inp.get("search_query", original_query)
                if cf := inp.get("classification_filter"):
                    filters.append(
                        ClassificationFilter(comparison_operator=cf["comparison_operator"], value=cf["value"])
                    )
                if catf := inp.get("category_filter"):
                    filters.append(
                        CategoryFilter(comparison_operator=catf["comparison_operator"], value=catf["value"])
                    )
        return search_query, filters

    async def prepare_context(self) -> tuple[list[CapabilityPublic], list[ThoughtStep]]:
        user_query = f"Find search results for user query: {self.chat_params.original_user_query}"
        messages_for_query = list(self.chat_params.past_messages) + [{"role": "user", "content": user_query}]

        response = await self.anthropic_chat_client.messages.create(
            model=self.chat_model,
            max_tokens=512,
            system=self.query_prompt_template,
            tools=[build_search_tool()],
            tool_choice={"type": "tool", "name": "search_database"},
            messages=messages_for_query,
        )

        search_query, filters = self._extract_search_arguments(user_query, response)

        results = await self.searcher.search_and_embed(
            search_query,
            top=self.chat_params.top,
            enable_vector_search=self.chat_params.enable_vector_search,
            enable_text_search=self.chat_params.enable_text_search,
            filters=filters,
        )
        items = [CapabilityPublic.model_validate(item.to_dict()) for item in results]

        thoughts = [
            ThoughtStep(
                title="Prompt to generate search arguments",
                description=[{"role": "system", "content": self.query_prompt_template}] + messages_for_query,
                props=self.model_for_thoughts,
            ),
            ThoughtStep(
                title="Search using generated search arguments",
                description=search_query,
                props={
                    "top": self.chat_params.top,
                    "vector_search": self.chat_params.enable_vector_search,
                    "text_search": self.chat_params.enable_text_search,
                    "filters": filters,
                },
            ),
            ThoughtStep(title="Search results", description=items),
        ]
        return items, thoughts

    async def answer(
        self,
        items: list[CapabilityPublic],
        earlier_thoughts: list[ThoughtStep],
    ) -> RetrievalResponse:
        rag_content = self.prepare_rag_request(self.chat_params.original_user_query, items)
        messages = list(self.chat_params.past_messages) + [{"role": "user", "content": rag_content}]

        response = await self.anthropic_chat_client.messages.create(
            model=self.chat_model,
            max_tokens=self.chat_params.response_token_limit,
            temperature=self.chat_params.temperature,
            system=self.chat_params.prompt_template,
            messages=messages,
        )
        output_text = response.content[0].text

        return RetrievalResponse(
            output_text=output_text,
            context=RAGContext(
                data_points={item.id: item for item in items},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[{"role": "system", "content": self.chat_params.prompt_template}] + messages,
                        props=self.model_for_thoughts,
                    )
                ],
            ),
        )

    async def answer_stream(
        self,
        items: list[CapabilityPublic],
        earlier_thoughts: list[ThoughtStep],
    ) -> AsyncGenerator[RetrievalResponseDelta, None]:
        rag_content = self.prepare_rag_request(self.chat_params.original_user_query, items)
        messages = list(self.chat_params.past_messages) + [{"role": "user", "content": rag_content}]

        yield RetrievalResponseDelta(
            type="response.context",
            context=RAGContext(
                data_points={item.id: item for item in items},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[{"role": "system", "content": self.chat_params.prompt_template}] + messages,
                        props=self.model_for_thoughts,
                    )
                ],
            ),
        )

        async with self.anthropic_chat_client.messages.stream(
            model=self.chat_model,
            max_tokens=self.chat_params.response_token_limit,
            temperature=self.chat_params.temperature,
            system=self.chat_params.prompt_template,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield RetrievalResponseDelta(type="response.output_text.delta", delta=text)
