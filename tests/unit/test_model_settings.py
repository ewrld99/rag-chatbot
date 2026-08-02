import pytest

from app.db.models import SystemSetting
from app.services.settings_service import SettingsService, seed_default_settings


def test_generation_model_settings_are_seeded(db_session):
    seed_default_settings(db_session)
    service = SettingsService(db_session)

    assert service.generation_default_model == "llama3.2:3b"
    assert service.generation_allowed_models == [
        "llama3.2:3b",
        "gemma3:4b",
        "qwen3.5:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    assert service.generation_answer_model_order == [
        "llama3.2:3b",
        "gemma3:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen3.5:4b",
    ]
    assert service.generation_utility_model_order == [
        "qwen2.5:1.5b",
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]
    utility_setting = (
        db_session.query(SystemSetting)
        .filter_by(key="generation_utility_model_order")
        .one()
    )
    assert utility_setting.description == (
        "Fallback priority for intent classification, query rewriting, "
        "clarification checks, and grounding verification"
    )
    assert service.generation_user_selection_enabled is True


def test_generation_settings_reject_models_outside_catalog(db_session):
    seed_default_settings(db_session)
    service = SettingsService(db_session)

    with pytest.raises(ValueError, match="unsupported"):
        service.update(
            "generation_allowed_models",
            "openai/gpt-oss-120b",
        )


def test_generation_settings_reject_gemini_for_utility_work(db_session):
    seed_default_settings(db_session)
    service = SettingsService(db_session)

    with pytest.raises(ValueError, match="utility-capable models"):
        service.update(
            "generation_utility_model_order",
            "llama-3.1-8b-instant,gemini-3.6-flash",
        )


def test_generation_settings_accept_small_qwen_for_utility_work_only(db_session):
    seed_default_settings(db_session)
    service = SettingsService(db_session)

    service.update(
        "generation_utility_model_order",
        "qwen2.5:1.5b,gemma3:4b",
    )

    assert SettingsService(db_session).generation_utility_model_order == [
        "qwen2.5:1.5b",
        "gemma3:4b",
    ]

    with pytest.raises(ValueError, match="utility-capable models"):
        service.update(
            "generation_utility_model_order",
            "qwen2.5:1.5b,qwen3.5:4b",
        )

    with pytest.raises(ValueError, match="unsupported"):
        service.update(
            "generation_allowed_models",
            "gemma3:4b,qwen2.5:1.5b",
        )


def test_seed_upgrades_known_legacy_generation_model_lists(db_session):
    db_session.add_all(
        [
            SystemSetting(
                key="generation_default_model",
                value="openai/gpt-oss-120b",
                description="legacy",
                category="generation",
            ),
            SystemSetting(
                key="generation_allowed_models",
                value="llama-3.3-70b-versatile,openai/gpt-oss-120b",
                description="legacy",
                category="generation",
            ),
            SystemSetting(
                key="generation_answer_model_order",
                value="llama-3.3-70b-versatile,openai/gpt-oss-120b",
                description="legacy",
                category="generation",
            ),
        ]
    )
    db_session.commit()

    seed_default_settings(db_session)
    service = SettingsService(db_session)

    assert service.generation_default_model == "llama3.2:3b"
    assert service.generation_allowed_models == [
        "llama3.2:3b",
        "gemma3:4b",
        "qwen3.5:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    assert service.generation_answer_model_order == [
        "llama3.2:3b",
        "gemma3:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen3.5:4b",
    ]


def test_seed_prepends_local_default_to_custom_generation_allowlist(db_session):
    custom_allowlist = "llama-3.3-70b-versatile,llama-3.1-8b-instant"
    db_session.add(
        SystemSetting(
            key="generation_allowed_models",
            value=custom_allowlist,
            description="custom",
            category="generation",
        )
    )
    db_session.commit()

    seed_default_settings(db_session)

    assert SettingsService(db_session).generation_allowed_models == [
        "llama3.2:3b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]


def test_seed_preserves_supported_custom_generation_default_after_upgrade(db_session):
    db_session.add_all(
        [
            SystemSetting(
                key="generation_default_model",
                value="gemma3:4b",
                description="custom",
                category="generation",
            ),
            SystemSetting(
                key="generation_allowed_models",
                value=(
                    "llama3.2:3b,gemma3:4b,qwen3.5:4b,gemini-3.6-flash,"
                    "llama-3.3-70b-versatile,llama-3.1-8b-instant"
                ),
                description="current",
                category="generation",
            ),
            SystemSetting(
                key="generation_answer_model_order",
                value=(
                    "llama3.2:3b,gemma3:4b,gemini-3.6-flash,"
                    "llama-3.3-70b-versatile,llama-3.1-8b-instant,qwen3.5:4b"
                ),
                description="current",
                category="generation",
            ),
        ]
    )
    db_session.commit()

    seed_default_settings(db_session)

    assert SettingsService(db_session).generation_default_model == "gemma3:4b"


def test_seed_moves_qwen35_to_last_in_known_answer_order(db_session):
    db_session.add(
        SystemSetting(
            key="generation_answer_model_order",
            value=(
                "gemma3:4b,qwen3.5:4b,gemini-3.6-flash,"
                "llama-3.3-70b-versatile,llama-3.1-8b-instant"
            ),
            description="old local order",
            category="generation",
        )
    )
    db_session.commit()

    seed_default_settings(db_session)

    assert SettingsService(db_session).generation_answer_model_order == [
        "llama3.2:3b",
        "gemma3:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen3.5:4b",
    ]


def test_seed_removes_qwen_from_generation_model_lists(db_session):
    db_session.add_all(
        [
            SystemSetting(
                key="generation_allowed_models",
                value=(
                    "llama-3.3-70b-versatile,qwen/qwen3.6-27b,"
                    "llama-3.1-8b-instant"
                ),
                description="old",
                category="generation",
            ),
            SystemSetting(
                key="generation_utility_model_order",
                value="llama-3.1-8b-instant,qwen/qwen3.6-27b",
                description="old",
                category="generation",
            ),
        ]
    )
    db_session.commit()

    seed_default_settings(db_session)
    service = SettingsService(db_session)

    assert service.generation_allowed_models == [
        "llama3.2:3b",
        "gemma3:4b",
        "qwen3.5:4b",
        "gemini-3.6-flash",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    assert service.generation_utility_model_order == [
        "qwen2.5:1.5b",
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]


def test_seed_migrates_known_utility_defaults_once(db_session):
    db_session.add(
        SystemSetting(
            key="generation_utility_model_order",
            value="gemma3:4b,llama-3.1-8b-instant,llama-3.3-70b-versatile",
            description="old default",
            category="generation",
        )
    )
    db_session.commit()

    seed_default_settings(db_session)
    service = SettingsService(db_session)

    assert service.generation_utility_model_order == [
        "qwen2.5:1.5b",
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]

    seed_default_settings(db_session)
    assert SettingsService(db_session).generation_utility_model_order == [
        "qwen2.5:1.5b",
        "gemma3:4b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]


def test_seed_preserves_custom_groq_utility_order(db_session):
    db_session.add(
        SystemSetting(
            key="generation_utility_model_order",
            value="llama-3.3-70b-versatile,llama-3.1-8b-instant",
            description="custom",
            category="generation",
        )
    )
    db_session.commit()

    seed_default_settings(db_session)

    assert SettingsService(db_session).generation_utility_model_order == [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
