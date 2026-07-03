import pytest
from app.services.settings_service import SettingsService
from app.db.models import SystemSetting, AuditLog

def test_settings_service_loads_and_caches(db_session):
    # Insert a fake setting directly
    setting = SystemSetting(key="chunk_size", value="800")
    db_session.add(setting)
    db_session.commit()

    svc = SettingsService(db_session)
    
    # Assert property reading works (should fetch from DB)
    assert svc.chunk_size == 800
    
    # Modifying DB externally shouldn't affect cache immediately
    setting.value = "900"
    db_session.commit()
    assert svc.chunk_size == 800  # Still cached

def test_settings_service_update_clears_cache_and_audits(db_session):
    # Prepare
    db_session.add(SystemSetting(key="chunk_size", value="800"))
    db_session.commit()
    
    svc = SettingsService(db_session)
    assert svc.chunk_size == 800
    
    # Update through service
    svc.update("chunk_size", "1200", admin_username="testadmin")
    
    # Assert cache cleared & property reflects new value
    assert svc.chunk_size == 1200
    
    # Assert audit log was created
    audit = db_session.query(AuditLog).filter_by(setting_key="chunk_size").first()
    assert audit is not None
    assert audit.admin == "testadmin"
    assert audit.old_value == "800"
    assert audit.new_value == "1200"

def test_settings_service_validation(db_session):
    db_session.add(SystemSetting(key="chunk_size", value="800"))
    db_session.commit()
    
    svc = SettingsService(db_session)
    
    # Out of bounds
    with pytest.raises(ValueError):
        svc.update("chunk_size", "400") # min is 500
        
    with pytest.raises(ValueError):
        svc.update("chunk_size", "3000") # max is 2000
        
    # Non-integer
    with pytest.raises(ValueError):
        svc.update("chunk_size", "abc")
