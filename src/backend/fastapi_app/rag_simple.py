from collections.abc import AsyncGenerator
from typing import Optional

import anthropic

from fastapi_app.api_models import (
    CapabilityPublic,
    ChatRequestOverrides,
    RAGContext,
    RetrievalResponse,
    RetrievalResponseDelta,
    ThoughtStep,
)
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import RAGChatBase


class SimpleRAGChat(RAGChatBase):
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
        self.chat_params = self.get_chat_params(messages, overrides)
        self.chat_model = chat_model
        self.model_for_thoughts = {"model": chat_model}

    async def prepare_context(self) -> tuple[list[CapabilityPublic], list[ThoughtStep]]:
        results = await self.searcher.search_and_embed(
            self.chat_params.original_user_query,
            top=self.chat_params.top,
            enable_vector_search=self.chat_params.enable_vector_search,
            enable_text_search=self.chat_params.enable_text_search,
        )
        items = [CapabilityPublic.model_validate(item.to_dict()) for item in results]
        thoughts = [
            ThoughtStep(
                title="Search query for database",
                description=self.chat_params.original_user_query,
                props={
                    "top": self.chat_params.top,
                    "vector_search": self.chat_params.enable_vector_search,
                    "text_search": self.chat_params.enable_text_search,
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
