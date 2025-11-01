# AIVA Converters Plugin Test Configuration
import pytest
import tempfile
import os
import json
from pathlib import Path
from typing import Dict, Any

# Test configuration
pytest_plugins = ["pytest_benchmark"]

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_output_dir():
    """Provide temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def sample_schemas(test_data_dir):
    """Load sample schemas for testing"""
    schemas_dir = test_data_dir / "schemas"
    if not schemas_dir.exists():
        schemas_dir.mkdir(parents=True)
        _create_sample_schemas(schemas_dir)
    
    return {
        schema.stem: schema.read_text()
        for schema in schemas_dir.glob("*.py")
    }

@pytest.fixture
def sample_sarif_data():
    """Provide sample SARIF data for testing"""
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "AIVA Security Scanner",
                    "version": "1.0.0"
                }
            },
            "results": [{
                "ruleId": "VULN-001",
                "message": {"text": "SQL Injection vulnerability"},
                "level": "error",
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": "/src/login.py"},
                        "region": {"startLine": 42}
                    }
                }]
            }]
        }]
    }

@pytest.fixture
def converter_instance():
    """Provide a configured converter instance"""
    from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodegen
    return SchemaCodegen()

@pytest.fixture
def validator_instance():
    """Provide a configured validator instance"""
    from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator
    return CrossLanguageValidator()

def _create_sample_schemas(schemas_dir: Path):
    """Create sample schema files for testing"""
    
    # Security scan schema
    security_scan_schema = '''
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ScanStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class VulnerabilityLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VulnerabilityFinding(BaseModel):
    id: str = Field(description="Unique finding identifier")
    title: str = Field(description="Vulnerability title")
    level: VulnerabilityLevel = Field(description="Severity level")
    description: Optional[str] = None
    location: Optional[str] = None
    cwe_id: Optional[int] = None
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0)

class ScanResult(BaseModel):
    scan_id: str = Field(description="Unique scan identifier")
    status: ScanStatus = Field(description="Current scan status")
    start_time: datetime = Field(description="Scan start timestamp")
    end_time: Optional[datetime] = None
    findings: List[VulnerabilityFinding] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
'''
    
    # Simple user schema
    user_schema = '''
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: int = Field(description="User ID")
    username: str = Field(description="Username", min_length=3, max_length=50)
    email: EmailStr = Field(description="Email address")
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(description="Creation timestamp")
'''
    
    # Write sample schemas
    (schemas_dir / "security_scan.py").write_text(security_scan_schema)
    (schemas_dir / "user.py").write_text(user_schema)
    
    # Create expected outputs directory
    expected_dir = schemas_dir.parent / "expected_outputs"
    expected_dir.mkdir(exist_ok=True)
    
    # TypeScript expected output for security scan
    ts_expected = '''
export enum ScanStatus {
  PENDING = "pending",
  RUNNING = "running", 
  COMPLETED = "completed",
  FAILED = "failed"
}

export enum VulnerabilityLevel {
  INFO = "info",
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical"
}

export interface VulnerabilityFinding {
  id: string;
  title: string;
  level: VulnerabilityLevel;
  description?: string | null;
  location?: string | null;
  cwe_id?: number | null;
  cvss_score?: number | null;
}

export interface ScanResult {
  scan_id: string;
  status: ScanStatus;
  start_time: string;
  end_time?: string | null;
  findings: VulnerabilityFinding[];
  summary: Record<string, number>;
}
'''
    (expected_dir / "security_scan.ts").write_text(ts_expected)

@pytest.fixture
def mock_language_tools():
    """Mock external language tools for testing"""
    import subprocess
    from unittest.mock import patch, MagicMock
    
    with patch('subprocess.run') as mock_run:
        # Mock successful TypeScript compilation
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock_run

# Test data samples
@pytest.fixture
def sample_vulnerability_data():
    """Sample vulnerability data for testing conversions"""
    return {
        "id": "VULN-TEST-001",
        "title": "Test SQL Injection",
        "level": "high",
        "description": "Test vulnerability for unit testing",
        "location": "test.py:123",
        "cwe_id": 89,
        "cvss_score": 8.5
    }

@pytest.fixture
def sample_scan_result():
    """Sample scan result data"""
    return {
        "scan_id": "SCAN-TEST-001",
        "status": "completed",
        "start_time": "2024-11-02T10:30:00Z",
        "end_time": "2024-11-02T10:35:00Z",
        "findings": [],
        "summary": {"total": 0, "critical": 0, "high": 0}
    }

# Performance test configuration
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks"""
    return {
        "min_rounds": 5,
        "max_time": 10.0,
        "warmup": True,
        "warmup_iterations": 2
    }

# Helper functions for tests
def assert_valid_python_code(code: str):
    """Assert that generated Python code is syntactically valid"""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Generated Python code is not syntactically valid: {e}")

def assert_contains_patterns(content: str, patterns: List[str]):
    """Assert that content contains all specified patterns"""
    for pattern in patterns:
        assert pattern in content, f"Pattern '{pattern}' not found in content"

def assert_file_exists_and_not_empty(file_path: Path):
    """Assert that file exists and is not empty"""
    assert file_path.exists(), f"File {file_path} does not exist"
    assert file_path.stat().st_size > 0, f"File {file_path} is empty"

# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)