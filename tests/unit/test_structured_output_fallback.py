import asyncio
from types import SimpleNamespace

from app.services.generation_resilience import GenerationUnavailableError
from app.services.generation_service import GenerationService
from app.services.generation_service import _GroundedClaimsPayload
from app.services.model_failover import ModelExecutionResult


class DirectFailover:
    def execute(self, _operation, preference, callback):
        return ModelExecutionResult(
            value=callback("test-model"),
            requested_model=preference or "auto",
            selected_model="test-model",
            fallback_used=False,
        )

    async def execute_async(self, _operation, preference, callback):
        return ModelExecutionResult(
            value=await callback("test-model"),
            requested_model=preference or "auto",
            selected_model="test-model",
            fallback_used=False,
        )


class TwoModelAsyncFailover:
    def __init__(self, models=None):
        self.models = models or ["bad-json-model", "good-json-model"]

    async def execute_async(self, _operation, preference, callback):
        errors = []
        for index, model in enumerate(self.models):
            try:
                return ModelExecutionResult(
                    value=await callback(model),
                    requested_model=preference or "auto",
                    selected_model=model,
                    fallback_used=index > 0,
                )
            except Exception as exc:
                errors.append(exc)
        raise GenerationUnavailableError() from errors[-1]


class SyncCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError(
                "Failed to generate JSON. See 'failed_generation' for more details."
            )
        message = SimpleNamespace(content='{"coverage":"none","answer":"","claims":[]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class EmptyNativeSyncCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            ""
            if "response_format" in kwargs
            else '{"coverage":"none","answer":"","claims":[]}'
        )
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AsyncTextStream:
    def __init__(self, content='{"coverage":"none","answer":"","claims":[]}'):
        self.content = content

    def __aiter__(self):
        async def chunks():
            delta = SimpleNamespace(
                content=self.content
            )
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        return chunks()


class AsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError(
                "Failed to validate JSON. See 'failed_generation' for more details."
            )
        return AsyncTextStream()


class EmptyNativeAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            ""
            if "response_format" in kwargs
            else '{"coverage":"none","answer":"","claims":[]}'
        )
        return AsyncTextStream(content)


class MalformedThenValidAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            "<think>I found the answer but did not return JSON.</think>"
            if kwargs["model"] in {"bad-json-model", "qwen3.5:4b"}
            else '{"coverage":"none","answer":"","claims":[]}'
        )
        return AsyncTextStream(content)


class NativeMalformedThenPromptValidAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = (
            "<think>I found the answer but did not return JSON.</think>"
            if "response_format" in kwargs
            else '{"coverage":"none","answer":"","claims":[]}'
        )
        return AsyncTextStream(content)


class NamedModelAsyncFailover:
    def __init__(self, model):
        self.model = model

    async def execute_async(self, _operation, preference, callback):
        return ModelExecutionResult(
            value=await callback(self.model),
            requested_model=preference or "auto",
            selected_model=self.model,
            fallback_used=False,
        )


class GeminiParsedAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(
            content='{"coverage":"none","answer":"","claims":[]}'
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_generator():
    generator = object.__new__(GenerationService)
    generator.model_failover = DirectFailover()
    generator.model_preference = "auto"
    generator._last_model_execution = None
    return generator


def test_sync_completion_retries_same_model_without_native_json_mode():
    completions = SyncCompletions()
    generator = make_generator()
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    response = generator.create_completion(
        "document_answer",
        messages=[{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
    )

    assert response.choices[0].message.content.startswith("{")
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in completions.calls[1]
    assert completions.calls[1]["model"] == "test-model"


def test_sync_completion_retries_empty_native_json_response():
    completions = EmptyNativeSyncCompletions()
    generator = make_generator()
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    response = generator.create_completion(
        "document_answer",
        messages=[{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
    )

    assert response.choices[0].message.content.startswith("{")
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]


def test_stream_completion_retries_same_model_without_native_json_mode():
    completions = AsyncCompletions()
    generator = make_generator()
    generator.async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = asyncio.run(
        generator._stream_text(
            "document_answer_stream",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]


def test_stream_completion_rejects_non_json_structured_text_and_fails_over():
    completions = MalformedThenValidAsyncCompletions()
    generator = make_generator()
    generator.model_failover = TwoModelAsyncFailover()
    generator.async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = asyncio.run(
        generator._stream_text(
            "document_answer_stream",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert [call["model"] for call in completions.calls] == [
        "bad-json-model",
        "bad-json-model",
        "good-json-model",
    ]
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
    assert "response_format" in completions.calls[2]
    assert generator._last_model_execution.selected_model == "good-json-model"
    assert generator._last_model_execution.fallback_used is True
    assert completions.calls[2]["stream"] is True


def test_qwen35_non_json_stream_fails_over_without_repeating_slow_request():
    completions = MalformedThenValidAsyncCompletions()
    generator = make_generator()
    generator.model_failover = TwoModelAsyncFailover(
        ["qwen3.5:4b", "good-json-model"]
    )
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    generator.async_client = mock_client
    generator.async_ollama_client = mock_client

    result = asyncio.run(
        generator._stream_text(
            "document_answer_stream",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert [call["model"] for call in completions.calls] == [
        "qwen3.5:4b",
        "good-json-model",
    ]


def test_stream_completion_recovers_non_json_on_same_model():
    completions = NativeMalformedThenPromptValidAsyncCompletions()
    generator = make_generator()
    generator.async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = asyncio.run(
        generator._stream_text(
            "document_answer_stream",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]


def test_lightweight_repair_caps_output_tokens():
    completions = NativeMalformedThenPromptValidAsyncCompletions()
    generator = make_generator()
    generator.model_failover = NamedModelAsyncFailover("llama-3.1-8b-instant")
    generator.async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = asyncio.run(
        generator._stream_text(
            "document_answer_repair_stream",
            messages=[{"role": "user", "content": "Repair JSON."}],
            max_tokens=950,
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert all(call["max_tokens"] == 500 for call in completions.calls)


def test_gemini_stream_uses_schema_parse_without_deprecated_parameters():
    completions = GeminiParsedAsyncCompletions()
    generator = make_generator()
    generator.model_failover = NamedModelAsyncFailover("gemini-3.6-flash")
    generator.async_gemini_client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
    )

    result = asyncio.run(
        generator._stream_text(
            "document_answer_stream",
            messages=[{"role": "user", "content": "Return JSON."}],
            temperature=0.0,
            max_tokens=750,
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert len(completions.calls) == 1
    assert completions.calls[0]["response_format"] is _GroundedClaimsPayload
    assert completions.calls[0]["reasoning_effort"] == "low"
    assert "temperature" not in completions.calls[0]
    assert "max_tokens" not in completions.calls[0]


def test_stream_completion_retries_empty_native_json_response():
    completions = EmptyNativeAsyncCompletions()
    generator = make_generator()
    generator.async_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = asyncio.run(
        generator._stream_text(
            "document_answer_stream",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_format={"type": "json_object"},
        )
    )

    assert result.startswith("{")
    assert len(completions.calls) == 2
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]
