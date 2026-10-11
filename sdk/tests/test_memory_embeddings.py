"""Memory embeddings use the normalized gateway instead of direct provider HTTP."""

import json
from typing import Any

import httpx
import pytest
from kernel.provider_contracts import EmbeddingRequest, EmbeddingResult, ProviderGatewayError
from openai import AsyncOpenAI
from sdk.memory.embeddings import OpenAIEmbeddingProvider
from sdk.providers.openai_gateway import OpenAICompatibleGateway


@pytest.mark.asyncio
async def test_memory_embeddings_dispatch_through_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[EmbeddingRequest] = []

    async def embed(self: OpenAICompatibleGateway, request: EmbeddingRequest) -> EmbeddingResult:
        requests.append(request)
        return EmbeddingResult(
            embeddings=[[0.1, 0.2]], provider=request.provider, model_id=request.model_id
        )

    monkeypatch.setattr(OpenAICompatibleGateway, "embed", embed)
    provider = OpenAIEmbeddingProvider(
        model_id="embed-model", endpoint="http://local/v1", dimension=2
    )
    assert await provider.embed(["hello"]) == [[0.1, 0.2]]
    assert requests == [
        EmbeddingRequest(input_texts=["hello"], provider="openai", model_id="embed-model")
    ]
    assert await provider.embed([]) == []
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["/v1", "/v1/", "/v1/embeddings", "/v1/embeddings/"])
async def test_embedding_urls_auth_and_vectors(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    captured: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    def client(**kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )

    monkeypatch.setattr("sdk.memory.embeddings.AsyncOpenAI", client)
    provider = OpenAIEmbeddingProvider(
        model_id="embed-model",
        endpoint=f"http://local{suffix}",
        dimension=2,
        api_key="test-key",
    )
    assert await provider.embed(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert captured[0].url.path == "/v1/embeddings"
    assert captured[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(captured[0].content)["input"] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [[], [[0.1]], [[0.1, 0.2], [0.3, 0.4]]])
async def test_embedding_shape_errors_propagate_to_logged_store_boundary(
    monkeypatch: pytest.MonkeyPatch, vectors: list[list[float]]
) -> None:
    async def embed(self: OpenAICompatibleGateway, request: EmbeddingRequest) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=vectors, provider=request.provider, model_id=request.model_id
        )

    monkeypatch.setattr(OpenAICompatibleGateway, "embed", embed)
    provider = OpenAIEmbeddingProvider(
        model_id="embed-model", endpoint="http://local/v1", dimension=2
    )
    with pytest.raises(ValueError, match="Embedding response"):
        await provider.embed(["hello"])


@pytest.mark.asyncio
async def test_embedding_http_failure_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "quota exhausted"}})

    def client(**kwargs: Any) -> AsyncOpenAI:
        return AsyncOpenAI(
            **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )

    monkeypatch.setattr("sdk.memory.embeddings.AsyncOpenAI", client)
    provider = OpenAIEmbeddingProvider(model_id="embed-model", endpoint="http://local/v1")
    with pytest.raises(ProviderGatewayError, match="quota exhausted"):
        await provider.embed(["hello"])
