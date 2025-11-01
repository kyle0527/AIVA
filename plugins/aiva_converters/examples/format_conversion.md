# Format Conversion Examples

## 📋 Overview

This guide demonstrates converting between different data formats and standards using the AIVA Converters Plugin.

## 🔄 Supported Format Conversions

### SARIF ↔ Custom Formats
Security Analysis Results Interchange Format conversions for vulnerability scanners.

### JSON ↔ YAML ↔ TOML
Configuration and data format conversions.

### Task Formats
Convert between different task runner configurations (VS Code tasks, GitHub Actions, etc.).

## 🛡️ SARIF Conversions

### Custom Scanner to SARIF
```python
# Convert custom vulnerability scan results to SARIF format
from plugins.aiva_converters.core.sarif_converter import SARIFConverter

converter = SARIFConverter()

# Custom scan result
custom_result = {
    "findings": [
        {
            "id": "VULN-001",
            "title": "SQL Injection in login form",
            "severity": "high", 
            "file_path": "/src/login.py",
            "line_number": 45,
            "description": "User input not properly sanitized",
            "remediation": "Use parameterized queries"
        }
    ],
    "metadata": {
        "scanner": "AIVA Security Scanner",
        "version": "2.1.0",
        "scan_time": "2024-11-02T10:30:00Z"
    }
}

# Convert to SARIF
sarif_result = converter.custom_to_sarif(custom_result)
```

#### Generated SARIF Output
```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "AIVA Security Scanner",
          "version": "2.1.0",
          "informationUri": "https://aiva.tools"
        }
      },
      "results": [
        {
          "ruleId": "VULN-001",
          "message": {
            "text": "SQL Injection in login form"
          },
          "level": "error",
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "/src/login.py"
                },
                "region": {
                  "startLine": 45
                }
              }
            }
          ],
          "properties": {
            "description": "User input not properly sanitized",
            "remediation": "Use parameterized queries"
          }
        }
      ]
    }
  ]
}
```

### SARIF to Custom Format
```python
# Convert SARIF results back to custom format
sarif_data = {...}  # SARIF JSON data
custom_result = converter.sarif_to_custom(sarif_data)

print(custom_result)
# Output:
# {
#   "findings": [...],
#   "metadata": {...}
# }
```

### Multiple Scanner Integration
```python
# Convert different scanner formats to unified SARIF
scanners = {
    "eslint": converter.eslint_to_sarif,
    "bandit": converter.bandit_to_sarif, 
    "semgrep": converter.semgrep_to_sarif,
    "custom": converter.custom_to_sarif
}

# Process multiple scan results
all_results = []
for scanner_name, scan_result in scan_results.items():
    if scanner_name in scanners:
        sarif_result = scanners[scanner_name](scan_result)
        all_results.append(sarif_result)

# Merge into single SARIF file
merged_sarif = converter.merge_sarif_results(all_results)
```

## 📝 Configuration Format Conversions

### JSON ↔ YAML
```python
from plugins.aiva_converters.core.config_converter import ConfigConverter

converter = ConfigConverter()

# JSON to YAML
json_config = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "credentials": {
            "username": "admin",
            "password": "${DB_PASSWORD}"
        }
    },
    "security": {
        "scan_types": ["vulnerability", "malware"],
        "max_concurrent_scans": 5
    }
}

yaml_config = converter.json_to_yaml(json_config)
```

#### Generated YAML
```yaml
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: ${DB_PASSWORD}

security:
  scan_types:
    - vulnerability
    - malware
  max_concurrent_scans: 5
```

### YAML ↔ TOML
```python
# YAML to TOML conversion
yaml_content = """
[database]
host = "localhost"
port = 5432

[database.credentials]
username = "admin"
password = "${DB_PASSWORD}"

[security]
scan_types = ["vulnerability", "malware"]
max_concurrent_scans = 5
"""

toml_config = converter.yaml_to_toml(yaml_content)
```

#### Generated TOML
```toml
[database]
host = "localhost"
port = 5432

[database.credentials]
username = "admin"
password = "${DB_PASSWORD}"

[security]
scan_types = ["vulnerability", "malware"]
max_concurrent_scans = 5
```

### Environment-Specific Configs
```python
# Generate environment-specific configurations
base_config = {...}  # Base configuration

environments = ["development", "staging", "production"]
for env in environments:
    env_config = converter.generate_env_config(base_config, env)
    converter.save_config(env_config, f"config.{env}.yaml")
```

## 🔧 Task Format Conversions

### VS Code Tasks ↔ GitHub Actions
```python
from plugins.aiva_converters.core.task_converter import TaskConverter

converter = TaskConverter()

# VS Code tasks.json
vscode_tasks = {
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Security Scan",
            "type": "shell",
            "command": "python",
            "args": ["-m", "aiva.scanner", "--target", "${input:scanTarget}"],
            "group": "test",
            "presentation": {
                "reveal": "always"
            }
        },
        {
            "label": "Build Documentation", 
            "type": "shell",
            "command": "sphinx-build",
            "args": ["-b", "html", "docs/", "docs/_build/"],
            "dependsOn": "Security Scan"
        }
    ]
}

# Convert to GitHub Actions
github_workflow = converter.vscode_to_github_actions(vscode_tasks)
```

#### Generated GitHub Actions Workflow
```yaml
name: AIVA Tasks
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Security Scan
        run: python -m aiva.scanner --target ${{ github.event.repository.html_url }}
        
  build-documentation:
    runs-on: ubuntu-latest
    needs: security-scan
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install sphinx
      - name: Build Documentation
        run: sphinx-build -b html docs/ docs/_build/
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/_build
```

### Makefile ↔ npm scripts
```python
# Convert Makefile to package.json scripts
makefile_content = """
install:
	pip install -r requirements.txt
	npm install

test:
	pytest tests/
	npm test

build:
	python setup.py build
	npm run build

deploy: build test
	docker build -t aiva:latest .
	docker push aiva:latest
"""

package_json_scripts = converter.makefile_to_npm_scripts(makefile_content)
```

#### Generated npm scripts
```json
{
  "scripts": {
    "install": "pip install -r requirements.txt && npm install",
    "test": "pytest tests/ && npm test",
    "build": "python setup.py build && npm run build",
    "deploy": "npm run build && npm run test && docker build -t aiva:latest . && docker push aiva:latest"
  }
}
```

## 🗂️ Data Structure Conversions

### CSV ↔ JSON
```python
from plugins.aiva_converters.core.data_converter import DataConverter

converter = DataConverter()

# CSV data
csv_data = """
id,title,severity,status,found_date
VULN-001,SQL Injection,high,open,2024-11-01
VULN-002,XSS Vulnerability,medium,fixed,2024-11-02
VULN-003,Path Traversal,high,open,2024-11-03
"""

# Convert to JSON
json_data = converter.csv_to_json(csv_data, {
    "id_field": "id",
    "date_fields": ["found_date"],
    "enum_fields": {"severity": ["low", "medium", "high", "critical"]}
})
```

#### Generated JSON
```json
[
  {
    "id": "VULN-001",
    "title": "SQL Injection", 
    "severity": "high",
    "status": "open",
    "found_date": "2024-11-01T00:00:00Z"
  },
  {
    "id": "VULN-002",
    "title": "XSS Vulnerability",
    "severity": "medium", 
    "status": "fixed",
    "found_date": "2024-11-02T00:00:00Z"
  },
  {
    "id": "VULN-003",
    "title": "Path Traversal",
    "severity": "high",
    "status": "open", 
    "found_date": "2024-11-03T00:00:00Z"
  }
]
```

### XML ↔ JSON
```python
# XML to JSON conversion
xml_data = """
<scan_report>
  <metadata>
    <scan_id>SCAN-123</scan_id>
    <target>https://example.com</target>
  </metadata>
  <findings>
    <finding id="VULN-001" severity="high">
      <title>SQL Injection</title>
      <location>login.php:45</location>
    </finding>
  </findings>
</scan_report>
"""

json_data = converter.xml_to_json(xml_data, {
    "preserve_attributes": True,
    "array_tags": ["finding"],
    "text_key": "value"
})
```

## 🔍 Advanced Conversion Features

### Schema-Aware Conversions
```python
# Use schema information for better conversions
from pydantic import BaseModel

class VulnerabilityFinding(BaseModel):
    id: str
    severity: int  # Will be validated as integer
    found_date: datetime  # Will be parsed as datetime

# Convert with schema validation
converter = DataConverter(schema=VulnerabilityFinding)
validated_data = converter.csv_to_json(csv_data, validate=True)
```

### Custom Conversion Rules
```python
# Define custom conversion rules
converter.add_rule("csv_to_json", {
    "date_format": "%Y-%m-%d",
    "null_values": ["", "NULL", "N/A"],
    "type_inference": True,
    "custom_transformers": {
        "severity": lambda x: {"low": 1, "medium": 5, "high": 8, "critical": 10}.get(x, 0)
    }
})
```

### Batch Processing
```bash
# Process multiple files
python plugins/aiva_converters/core/format_converter.py \
  --input-dir data/csv/ \
  --output-dir data/json/ \
  --from csv \
  --to json \
  --schema schemas/vulnerability.json
```

## 🧪 Testing Conversions

### Roundtrip Testing
```python
# Test conversion accuracy with roundtrip
original_data = {...}

# JSON → YAML → JSON
yaml_converted = converter.json_to_yaml(original_data)
roundtrip_data = converter.yaml_to_json(yaml_converted)

# Verify data integrity
assert original_data == roundtrip_data

# Test with fuzzing
import hypothesis

@hypothesis.given(hypothesis.strategies.dictionaries(
    hypothesis.strategies.text(), 
    hypothesis.strategies.integers()
))
def test_json_yaml_roundtrip(data):
    yaml_converted = converter.json_to_yaml(data)
    roundtrip_data = converter.yaml_to_json(yaml_converted)
    assert data == roundtrip_data
```

### Performance Testing
```python
# Benchmark conversion performance
import time
import json

large_data = {"items": [{"id": i, "data": f"item_{i}"} for i in range(10000)]}

start_time = time.time()
yaml_result = converter.json_to_yaml(large_data)
yaml_time = time.time() - start_time

start_time = time.time()
toml_result = converter.json_to_toml(large_data)
toml_time = time.time() - start_time

print(f"YAML conversion: {yaml_time:.3f}s")
print(f"TOML conversion: {toml_time:.3f}s")
```

---

**Example Updated**: November 2, 2025  
**Plugin Version**: 1.0.0