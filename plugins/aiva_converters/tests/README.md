# AIVA Converters Plugin - Test Suite

This directory contains comprehensive tests for the AIVA Converters Plugin functionality.

## 📁 Directory Structure

```
tests/
├── unit/
│   ├── test_schema_codegen.py       # Schema generation tests
│   ├── test_cross_language.py       # Cross-language validation tests
│   ├── test_format_conversion.py    # Format conversion tests
│   └── test_template_engine.py      # Template engine tests
├── integration/
│   ├── test_multi_language.py       # Multi-language integration tests
│   ├── test_api_compatibility.py    # API compatibility tests
│   └── test_roundtrip.py           # Roundtrip conversion tests
├── performance/
│   ├── benchmark_generation.py      # Performance benchmarks
│   └── stress_test.py              # Stress testing
├── fixtures/
│   ├── schemas/                     # Test schema files
│   ├── expected_outputs/            # Expected generation outputs
│   └── sample_data/                 # Sample test data
└── conftest.py                      # Pytest configuration
```

## 🧪 Running Tests

### All Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=plugins.aiva_converters --cov-report=html

# Run with verbose output
python -m pytest tests/ -v
```

### Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Performance benchmarks
python -m pytest tests/performance/ --benchmark-only
```

### Individual Test Files
```bash
# Schema generation tests
python -m pytest tests/unit/test_schema_codegen.py

# Cross-language validation tests
python -m pytest tests/unit/test_cross_language.py
```

## 📋 Test Categories

### Unit Tests
- **Schema Code Generation**: Test individual language code generators
- **Cross-Language Validation**: Test schema compatibility validation
- **Format Conversion**: Test format conversion functionality
- **Template Engine**: Test template rendering and customization

### Integration Tests
- **Multi-Language**: Test complete workflow across multiple languages
- **API Compatibility**: Test generated code API compatibility
- **Roundtrip Conversion**: Test data integrity through conversion chains

### Performance Tests
- **Generation Benchmarks**: Measure code generation performance
- **Memory Usage**: Monitor memory consumption during conversion
- **Stress Testing**: Test with large schemas and datasets

## 🎯 Test Configuration

### Environment Variables
```bash
# Test configuration
export AIVA_TEST_MODE=true
export AIVA_LOG_LEVEL=DEBUG
export AIVA_TEST_TIMEOUT=30

# Language tool paths (for integration tests)
export TYPESCRIPT_BIN=/usr/local/bin/tsc
export RUST_BIN=/usr/local/bin/cargo
export GO_BIN=/usr/local/bin/go
```

### Pytest Configuration (conftest.py)
```python
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_output_dir():
    """Provide temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_schemas(test_data_dir):
    """Load sample schemas for testing"""
    schemas_dir = test_data_dir / "schemas"
    return {
        schema.stem: schema.read_text()
        for schema in schemas_dir.glob("*.py")
    }
```

## 🔍 Test Examples

### Schema Generation Test
```python
def test_typescript_generation(temp_output_dir, sample_schemas):
    """Test TypeScript interface generation"""
    from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodegen
    
    codegen = SchemaCodegen()
    
    # Generate TypeScript from Python schema
    result = codegen.generate(
        schema_content=sample_schemas["security_scan"],
        target_language="typescript",
        output_dir=temp_output_dir
    )
    
    # Verify generation success
    assert result.success
    assert result.output_file.exists()
    
    # Verify generated content
    content = result.output_file.read_text()
    assert "export interface" in content
    assert "ScanResult" in content
```

### Cross-Language Validation Test
```python
def test_schema_compatibility():
    """Test schema compatibility across languages"""
    from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator
    
    validator = CrossLanguageValidator()
    
    # Test compatibility between Python and TypeScript
    result = validator.validate_compatibility(
        source_schema="SecurityScan",
        source_language="python", 
        target_language="typescript"
    )
    
    assert result.is_compatible
    assert result.compatibility_score >= 90
    assert len(result.issues) == 0
```

### Format Conversion Test
```python
def test_sarif_conversion():
    """Test SARIF format conversion"""
    from plugins.aiva_converters.core.sarif_converter import SARIFConverter
    
    converter = SARIFConverter()
    
    # Sample vulnerability data
    custom_data = {
        "findings": [
            {
                "id": "VULN-001",
                "title": "SQL Injection",
                "severity": "high",
                "file_path": "/src/login.py",
                "line_number": 42
            }
        ]
    }
    
    # Convert to SARIF
    sarif_result = converter.custom_to_sarif(custom_data)
    
    # Validate SARIF structure
    assert sarif_result["version"] == "2.1.0"
    assert len(sarif_result["runs"]) == 1
    assert len(sarif_result["runs"][0]["results"]) == 1
```

## 📊 Performance Testing

### Benchmark Configuration
```python
# tests/performance/benchmark_generation.py
import pytest
import time
from plugins.aiva_converters.core.schema_codegen_tool import SchemaCodegen

class TestPerformanceBenchmarks:
    
    @pytest.mark.benchmark
    def test_large_schema_generation_performance(self, benchmark):
        """Benchmark code generation with large schemas"""
        
        def generate_large_schema():
            # Create large schema with 100+ fields
            schema_content = self._create_large_schema(field_count=100)
            
            codegen = SchemaCodegen()
            return codegen.generate(
                schema_content=schema_content,
                target_language="typescript"
            )
        
        result = benchmark(generate_large_schema)
        assert result.success
    
    def test_memory_usage_during_generation(self):
        """Test memory usage during code generation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate multiple schemas
        codegen = SchemaCodegen()
        for i in range(50):
            schema_content = self._create_test_schema(f"TestSchema{i}")
            codegen.generate(schema_content, "typescript")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Assert memory increase is reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
```

## 🎯 Continuous Integration

### GitHub Actions Configuration
```yaml
name: AIVA Converters Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r plugins/aiva_converters/requirements.txt
        pip install pytest pytest-cov pytest-benchmark
    
    - name: Run tests
      run: |
        python -m pytest plugins/aiva_converters/tests/ \
          --cov=plugins.aiva_converters \
          --cov-report=xml \
          --benchmark-skip
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 🔧 Test Utilities

### Custom Assertions
```python
# tests/utils/assertions.py
def assert_valid_typescript(code_content):
    """Assert that generated TypeScript code is valid"""
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".ts", mode="w") as f:
        f.write(code_content)
        f.flush()
        
        # Run TypeScript compiler check
        result = subprocess.run(
            ["tsc", "--noEmit", f.name],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"TypeScript compilation failed: {result.stderr}"

def assert_schema_compatibility(schema1, schema2, min_score=90):
    """Assert that two schemas are compatible"""
    from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator
    
    validator = CrossLanguageValidator()
    result = validator.compare_schemas(schema1, schema2)
    
    assert result.compatibility_score >= min_score
    assert result.is_compatible
```

---

**Test Documentation Updated**: November 2, 2025  
**Plugin Version**: 1.0.0

---

[← 返回 AIVA Converters](../README.md) | [← 返回 Plugins 主目錄](../../README.md)