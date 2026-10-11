"""Embedding adapter for stores, using the SDK's OpenAI-compatible gateway."""

from kernel.provider_contracts import EmbeddingRequest
from openai import AsyncOpenAI
from sdk.providers.openai_gateway import OpenAICompatibleGateway


class OpenAIEmbeddingProvider:
    """Resolve one vector per text through an OpenAI-compatible embedding gateway.

    Accept a base endpoint or its full ``/embeddings`` URL. Provider failures
    propagate to the memory store's logged, best-effort embedding boundary.
    """

    def __init__(
        self,
        *,
        model_id: str,
        endpoint: str,
        api_key: str | None = None,
        dimension: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        self._model_id = model_id
        self._base_url = endpoint.rstrip("/").removesuffix("/embeddings") + "/"
        self._api_key = api_key or "not-required"
        self._dimension = dimension
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=0,
        ) as client:
            gateway = OpenAICompatibleGateway(client_factory=lambda: client)
            result = await gateway.embed(
                EmbeddingRequest(input_texts=texts, provider="openai", model_id=self._model_id)
            )
        if len(result.embeddings) != len(texts):
            raise ValueError(
                f"Embedding response returned {len(result.embeddings)} vectors "
                f"for {len(texts)} texts."
            )
        if any(len(vector) != self._dimension for vector in result.embeddings):
            raise ValueError(f"Embedding response does not match dimension {self._dimension}.")
        return result.embeddings
