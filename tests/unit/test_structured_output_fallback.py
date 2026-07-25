import asyncio
from types import SimpleNamespace

from app.services.generation_service import GenerationService
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
    assert completions.calls[1]["stream"] is True


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
