import pytest

from app.services.settings_service import SettingsService, seed_default_settings


def test_generation_model_settings_are_seeded(db_session):
    seed_default_settings(db_session)
    service = SettingsService(db_session)

    assert service.generation_default_model == "llama-3.3-70b-versatile"
    assert service.generation_allowed_models == [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ]
    assert service.generation_user_selection_enabled is True


def test_generation_settings_reject_models_outside_catalog(db_session):
    seed_default_settings(db_session)
    service = SettingsService(db_session)

    with pytest.raises(ValueError, match="unsupported"):
        service.update(
            "generation_allowed_models",
            "openai/gpt-oss-120b,unapproved/model",
        )
