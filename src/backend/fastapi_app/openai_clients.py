import logging
import os

import anthropic
import azure.identity.aio
import openai

logger = logging.getLogger("ragapp")


async def create_anthropic_chat_client() -> anthropic.AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required for chat.")
    logger.info("Setting up Anthropic client for chat")
    return anthropic.AsyncAnthropic(api_key=api_key)


async def create_openai_embed_client(
    azure_credential: azure.identity.aio.AzureDeveloperCliCredential
    | azure.identity.aio.ManagedIdentityCredential
    | None,
) -> openai.AsyncOpenAI:
    openai_embed_client: openai.AsyncOpenAI
    OPENAI_EMBED_HOST = os.getenv("OPENAI_EMBED_HOST")
    if OPENAI_EMBED_HOST == "azure":
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_deployment = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
        if api_key := os.getenv("AZURE_OPENAI_KEY"):
            logger.info(
                "Setting up Azure OpenAI client for embeddings using API key, endpoint %s, deployment %s",
                azure_endpoint,
                azure_deployment,
            )
            openai_embed_client = openai.AsyncOpenAI(
                base_url=f"{azure_endpoint.rstrip('/')}/openai/v1/",
                api_key=api_key,
            )
        elif azure_credential:
            logger.info(
                "Setting up Azure OpenAI client for embeddings using Azure Identity, endpoint %s, deployment %s",
                azure_endpoint,
                azure_deployment,
            )
            token_provider = azure.identity.aio.get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
            openai_embed_client = openai.AsyncOpenAI(
                base_url=f"{azure_endpoint.rstrip('/')}/openai/v1/",
                api_key=token_provider,
            )
        else:
            raise ValueError("Azure OpenAI client requires either an API key or Azure Identity credential.")
    elif OPENAI_EMBED_HOST == "ollama":
        logger.info("Setting up OpenAI client for embeddings using Ollama")
        openai_embed_client = openai.AsyncOpenAI(
            base_url=os.getenv("OLLAMA_ENDPOINT"),
            api_key="nokeyneeded",
        )
    else:
        logger.info("Setting up OpenAI client for embeddings using OpenAI.com API key")
        openai_embed_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAICOM_KEY"))
    return openai_embed_client
