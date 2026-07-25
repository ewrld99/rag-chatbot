import pytest

from app.services.jina_resilience import (
    JinaProviderCooldownError,
    is_jina_account_failure,
    jina_provider_circuit,
)


class FakeJinaBalanceError(RuntimeError):
    status_code = 403
    code = "AUTHZ_INSUFFICIENT_BALANCE"


@pytest.fixture(autouse=True)
def reset_jina_circuit():
    jina_provider_circuit.reset()
    yield
    jina_provider_circuit.reset()


def test_jina_balance_error_is_an_account_level_failure():
    error = FakeJinaBalanceError("Insufficient account balance.")

    assert is_jina_account_failure(error) is True


def test_jina_account_failure_opens_shared_cooldown(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "JINA_AUTH_COOLDOWN_SECONDS", 300.0)
    jina_provider_circuit.record_account_failure()

    with pytest.raises(JinaProviderCooldownError) as captured:
        jina_provider_circuit.before_call()

    assert 1 <= captured.value.retry_after <= 300
