import pytest
import sys
import importlib.util
from unittest.mock import patch, mock_open

# Inject mock for cryptography.fernet to bypass environment issue
import sys
from unittest.mock import MagicMock
sys.modules['cryptography'] = MagicMock()
sys.modules['cryptography.fernet'] = MagicMock()

from config.api_keys import APIKeyManager, get_api_key, set_api_key, has_api_key, validate_api_keys

@pytest.fixture
def mock_api_keys():
    # Mock the api_keys global object used by the convenience functions
    manager = APIKeyManager(key_file="dummy.json", encryption_key="dummy_key")
    manager.keys = {
        "jwt_secret": "test_jwt_secret",
        "api_secret_key": "test_api_secret"
    }
    with patch("config.api_keys.api_keys", manager):
        yield manager

def test_get_api_key(mock_api_keys):
    assert get_api_key("jwt_secret") == "test_jwt_secret"
    assert get_api_key("missing_key") is None
    assert get_api_key("missing_key", "default_value") == "default_value"

def test_set_api_key(mock_api_keys):
    set_api_key("new_key", "new_value")
    assert mock_api_keys.keys["new_key"] == "new_value"

def test_has_api_key(mock_api_keys):
    assert has_api_key("jwt_secret") is True
    assert has_api_key("missing_key") is False

def test_validate_api_keys(mock_api_keys):
    validation_results = validate_api_keys()

    assert "jwt_secret" in validation_results
    assert validation_results["jwt_secret"] is True

    assert "api_secret_key" in validation_results
    assert validation_results["api_secret_key"] is True

    assert "hackerone_credentials" in validation_results
    assert "bugcrowd_credentials" in validation_results

    assert validation_results["hackerone_credentials"] is False

    mock_api_keys.keys["hackerone_api_token"] = "token"
    mock_api_keys.keys["hackerone_username"] = "user"

    new_results = validate_api_keys()
    assert new_results["hackerone_credentials"] is True
