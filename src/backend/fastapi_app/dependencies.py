import logging
import os
from collections.abc import AsyncGenerator
from typing import Annotated, Optional

import anthropic
import azure.identity.aio
from fastapi import Depends, Request
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

logger = logging.getLogger("ragapp")


class OpenAIClient(BaseModel):
    client: AsyncOpenAI
    model_config = {"arbitrary_types_allowed": True}


class AnthropicClient(BaseModel):
    client: anthropic.AsyncAnthropic
    model_config = {"arbitrary_types_allowed": True}


class FastAPIAppContext(BaseModel):
    anthropic_chat_model: str
    openai_embed_model: str
    openai_embed_dimensions: Optional[int]
    openai_embed_deployment: Optional[str]
    embedding_column: str


async def common_parameters():
    OPENAI_EMBED_HOST = os.getenv("OPENAI_EMBED_HOST")
    if OPENAI_EMBED_HOST == "azure":
        openai_embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT") or "text-embedding-3-large"
        openai_embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL") or "text-embedding-3-large"
        openai_embed_dimensions = int(os.getenv("AZURE_OPENAI_EMBED_DIMENSIONS") or 1024)
        embedding_column = os.getenv("AZURE_OPENAI_EMBEDDING_COLUMN") or "embedding_3l"
    elif OPENAI_EMBED_HOST == "ollama":
        openai_embed_deployment = None
        openai_embed_model = os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text"
        openai_embed_dimensions = None
        embedding_column = os.getenv("OLLAMA_EMBEDDING_COLUMN") or "embedding_nomic"
    else:
        openai_embed_deployment = None
        openai_embed_model = os.getenv("OPENAICOM_EMBED_MODEL") or "text-embedding-3-large"
        openai_embed_dimensions = int(os.getenv("OPENAICOM_EMBED_DIMENSIONS", 1024))
        embedding_column = os.getenv("OPENAICOM_EMBEDDING_COLUMN") or "embedding_3l"

    anthropic_chat_model = os.getenv("ANTHROPIC_CHAT_MODEL") or "claude-opus-4-7"

    return FastAPIAppContext(
        anthropic_chat_model=anthropic_chat_model,
        openai_embed_model=openai_embed_model,
        openai_embed_dimensions=openai_embed_dimensions,
        openai_embed_deployment=openai_embed_deployment,
        embedding_column=embedding_column,
    )


async def get_azure_credential() -> (
    azure.identity.aio.AzureDeveloperCliCredential | azure.identity.aio.ManagedIdentityCredential
):
    azure_credential: azure.identity.aio.AzureDeveloperCliCredential | azure.identity.aio.ManagedIdentityCredential
    try:
        if client_id := os.getenv("APP_IDENTITY_ID"):
            logger.info("Using managed identity for client ID %s", client_id)
            azure_credential = azure.identity.aio.ManagedIdentityCredential(client_id=client_id)
        else:
            if tenant_id := os.getenv("AZURE_TENANT_ID"):
                logger.info("Authenticating to Azure using Azure Developer CLI Credential for tenant %s", tenant_id)
                azure_credential = azure.identity.aio.AzureDeveloperCliCredential(tenant_id=tenant_id)
            else:
                logger.info("Authenticating to Azure using Azure Developer CLI Credential")
                azure_credential = azure.identity.aio.AzureDeveloperCliCredential()
        return azure_credential
    except Exception as e:
        logger.warning("Failed to authenticate to Azure: %s", e)
        raise e


async def create_async_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def get_async_sessionmaker(
    request: Request,
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    yield request.state.sessionmaker


async def get_context(request: Request) -> FastAPIAppContext:
    return request.state.context


async def get_async_db_session(
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_async_sessionmaker)],
) -> AsyncGenerator[AsyncSession, None]:
    async with sessionmaker() as session:
        yield session


async def get_anthropic_chat_client(request: Request) -> AnthropicClient:
    return AnthropicClient(client=request.state.chat_client)


async def get_openai_embed_client(request: Request) -> OpenAIClient:
    return OpenAIClient(client=request.state.embed_client)


CommonDeps = Annotated[FastAPIAppContext, Depends(get_context)]
DBSession = Annotated[AsyncSession, Depends(get_async_db_session)]
ChatClient = Annotated[AnthropicClient, Depends(get_anthropic_chat_client)]
EmbeddingsClient = Annotated[OpenAIClient, Depends(get_openai_embed_client)]
