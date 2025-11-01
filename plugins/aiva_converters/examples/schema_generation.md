# Schema Generation Examples

## 📋 Overview

This guide demonstrates comprehensive schema generation capabilities across multiple languages using the AIVA Converters Plugin.

## 🎯 Multi-Language Schema Generation

### Source Schema (Python/Pydantic)
```python
# schemas/security_scan.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ScanType(str, Enum):
    VULNERABILITY = "vulnerability"
    MALWARE = "malware"
    NETWORK = "network"
    COMPLIANCE = "compliance"

class Finding(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(description="Unique finding identifier")
    type: ScanType = Field(description="Type of security finding")
    severity: int = Field(ge=1, le=10, description="Severity score (1-10)")
    title: str = Field(max_length=200, description="Finding title")
    description: Optional[str] = Field(None, description="Detailed description")
    affected_resources: List[str] = Field(default_factory=list)
    remediation: Optional[str] = Field(None, description="Remediation steps")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ScanReport(BaseModel):
    scan_id: str = Field(description="Unique scan identifier")
    target: str = Field(description="Scan target (URL, IP, etc.)")
    scan_type: ScanType = Field(description="Type of scan performed")
    start_time: datetime = Field(description="Scan start timestamp")
    end_time: Optional[datetime] = Field(None, description="Scan end timestamp")
    findings: List[Finding] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict, description="Finding counts by severity")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

## 🔧 Generation Commands

### All Languages at Once
```bash
# Generate schemas for all supported languages
python plugins/aiva_converters/core/schema_codegen_tool.py --all-languages

# Generate with validation
python plugins/aiva_converters/core/schema_codegen_tool.py --all-languages --validate
```

### Language-Specific Generation
```bash
# TypeScript interfaces
python plugins/aiva_converters/core/schema_codegen_tool.py --lang typescript

# Rust structs
python plugins/aiva_converters/core/schema_codegen_tool.py --lang rust

# Go structs
python plugins/aiva_converters/core/schema_codegen_tool.py --lang go

# C# classes
python plugins/aiva_converters/core/schema_codegen_tool.py --lang csharp
```

## 📝 Generated Output Examples

### TypeScript (interfaces/security_scan.ts)
```typescript
export enum ScanType {
  VULNERABILITY = "vulnerability",
  MALWARE = "malware",
  NETWORK = "network",
  COMPLIANCE = "compliance"
}

export interface Finding {
  /** Unique finding identifier */
  id: string;
  
  /** Type of security finding */
  type: ScanType;
  
  /** Severity score (1-10) */
  severity: number;
  
  /** Finding title */
  title: string;
  
  /** Detailed description */
  description?: string | null;
  
  /** List of affected resources */
  affected_resources: string[];
  
  /** Remediation steps */
  remediation?: string | null;
  
  /** Additional metadata */
  metadata: Record<string, any>;
}

export interface ScanReport {
  /** Unique scan identifier */
  scan_id: string;
  
  /** Scan target (URL, IP, etc.) */
  target: string;
  
  /** Type of scan performed */
  scan_type: ScanType;
  
  /** Scan start timestamp */
  start_time: string; // ISO 8601 format
  
  /** Scan end timestamp */
  end_time?: string | null; // ISO 8601 format
  
  /** List of findings */
  findings: Finding[];
  
  /** Finding counts by severity */
  summary: Record<string, number>;
  
  /** Additional metadata */
  metadata: Record<string, any>;
}
```

### Rust (structs/security_scan.rs)
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanType {
    Vulnerability,
    Malware,
    Network,
    Compliance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Unique finding identifier
    pub id: String,
    
    /// Type of security finding
    #[serde(rename = "type")]
    pub scan_type: ScanType,
    
    /// Severity score (1-10)
    pub severity: i32,
    
    /// Finding title
    pub title: String,
    
    /// Detailed description
    pub description: Option<String>,
    
    /// List of affected resources
    pub affected_resources: Vec<String>,
    
    /// Remediation steps
    pub remediation: Option<String>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    /// Unique scan identifier
    pub scan_id: String,
    
    /// Scan target (URL, IP, etc.)
    pub target: String,
    
    /// Type of scan performed
    pub scan_type: ScanType,
    
    /// Scan start timestamp
    pub start_time: DateTime<Utc>,
    
    /// Scan end timestamp
    pub end_time: Option<DateTime<Utc>>,
    
    /// List of findings
    pub findings: Vec<Finding>,
    
    /// Finding counts by severity
    pub summary: HashMap<String, i32>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}
```

### Go (models/security_scan.go)
```go
package models

import (
    "encoding/json"
    "time"
)

// ScanType represents the type of security scan
type ScanType string

const (
    ScanTypeVulnerability ScanType = "vulnerability"
    ScanTypeMalware      ScanType = "malware"
    ScanTypeNetwork      ScanType = "network"
    ScanTypeCompliance   ScanType = "compliance"
)

// Finding represents a security finding from a scan
type Finding struct {
    // Unique finding identifier
    ID string `json:"id" validate:"required"`
    
    // Type of security finding
    Type ScanType `json:"type" validate:"required"`
    
    // Severity score (1-10)
    Severity int `json:"severity" validate:"required,min=1,max=10"`
    
    // Finding title
    Title string `json:"title" validate:"required,max=200"`
    
    // Detailed description
    Description *string `json:"description,omitempty"`
    
    // List of affected resources
    AffectedResources []string `json:"affected_resources"`
    
    // Remediation steps
    Remediation *string `json:"remediation,omitempty"`
    
    // Additional metadata
    Metadata map[string]interface{} `json:"metadata"`
}

// ScanReport represents a complete scan report
type ScanReport struct {
    // Unique scan identifier
    ScanID string `json:"scan_id" validate:"required"`
    
    // Scan target (URL, IP, etc.)
    Target string `json:"target" validate:"required"`
    
    // Type of scan performed
    ScanType ScanType `json:"scan_type" validate:"required"`
    
    // Scan start timestamp
    StartTime time.Time `json:"start_time" validate:"required"`
    
    // Scan end timestamp
    EndTime *time.Time `json:"end_time,omitempty"`
    
    // List of findings
    Findings []Finding `json:"findings"`
    
    // Finding counts by severity
    Summary map[string]int `json:"summary"`
    
    // Additional metadata
    Metadata map[string]interface{} `json:"metadata"`
}

// Validate performs validation on the ScanReport
func (sr *ScanReport) Validate() error {
    // Implementation would use go-playground/validator
    return nil
}
```

### C# (Models/SecurityScan.cs)
```csharp
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace AIVA.Models
{
    /// <summary>
    /// Enumeration of supported scan types
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum ScanType
    {
        Vulnerability,
        Malware,
        Network,
        Compliance
    }

    /// <summary>
    /// Represents a security finding from a scan
    /// </summary>
    public class Finding
    {
        /// <summary>
        /// Unique finding identifier
        /// </summary>
        [Required]
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        /// <summary>
        /// Type of security finding
        /// </summary>
        [Required]
        [JsonPropertyName("type")]
        public ScanType Type { get; set; }

        /// <summary>
        /// Severity score (1-10)
        /// </summary>
        [Required]
        [Range(1, 10)]
        [JsonPropertyName("severity")]
        public int Severity { get; set; }

        /// <summary>
        /// Finding title
        /// </summary>
        [Required]
        [StringLength(200)]
        [JsonPropertyName("title")]
        public string Title { get; set; } = string.Empty;

        /// <summary>
        /// Detailed description
        /// </summary>
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        /// <summary>
        /// List of affected resources
        /// </summary>
        [JsonPropertyName("affected_resources")]
        public List<string> AffectedResources { get; set; } = new();

        /// <summary>
        /// Remediation steps
        /// </summary>
        [JsonPropertyName("remediation")]
        public string? Remediation { get; set; }

        /// <summary>
        /// Additional metadata
        /// </summary>
        [JsonPropertyName("metadata")]
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Represents a complete scan report
    /// </summary>
    public class ScanReport
    {
        /// <summary>
        /// Unique scan identifier
        /// </summary>
        [Required]
        [JsonPropertyName("scan_id")]
        public string ScanId { get; set; } = string.Empty;

        /// <summary>
        /// Scan target (URL, IP, etc.)
        /// </summary>
        [Required]
        [JsonPropertyName("target")]
        public string Target { get; set; } = string.Empty;

        /// <summary>
        /// Type of scan performed
        /// </summary>
        [Required]
        [JsonPropertyName("scan_type")]
        public ScanType ScanType { get; set; }

        /// <summary>
        /// Scan start timestamp
        /// </summary>
        [Required]
        [JsonPropertyName("start_time")]
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Scan end timestamp
        /// </summary>
        [JsonPropertyName("end_time")]
        public DateTime? EndTime { get; set; }

        /// <summary>
        /// List of findings
        /// </summary>
        [JsonPropertyName("findings")]
        public List<Finding> Findings { get; set; } = new();

        /// <summary>
        /// Finding counts by severity
        /// </summary>
        [JsonPropertyName("summary")]
        public Dictionary<string, int> Summary { get; set; } = new();

        /// <summary>
        /// Additional metadata
        /// </summary>
        [JsonPropertyName("metadata")]
        public Dictionary<string, object> Metadata { get; set; } = new();
    }
}
```

## 🔄 Cross-Language Validation

### Validation Script
```python
# Cross-language compatibility check
from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator

validator = CrossLanguageValidator()

# Validate schema compatibility across all languages
results = validator.validate_multi_language_schemas(
    schema_name="ScanReport",
    languages=["python", "typescript", "rust", "go", "csharp"]
)

for language, result in results.items():
    print(f"{language}: {result.compatibility_score}% compatible")
    if not result.is_compatible:
        print(f"  Issues: {result.issues}")
```

## 📊 Advanced Features

### Custom Templates
```bash
# Generate with custom templates
python plugins/aiva_converters/core/schema_codegen_tool.py \
  --lang typescript \
  --template plugins/aiva_converters/templates/custom_interface.j2
```

### Configuration Files
```yaml
# schema_config.yaml
generation:
  output_dir: "generated/"
  languages:
    - typescript
    - rust
    - go
  
validation:
  strict_mode: true
  check_compatibility: true
  
templates:
  typescript: "templates/typescript/interface.j2"
  rust: "templates/rust/struct.j2"
  go: "templates/go/model.j2"
```

### Batch Processing
```bash
# Process multiple schema files
python plugins/aiva_converters/core/schema_codegen_tool.py \
  --input schemas/*.py \
  --output generated/ \
  --config schema_config.yaml
```

## 🧪 Testing Generated Schemas

### Validation Tests
```python
# test_generated_schemas.py
import pytest
from plugins.aiva_converters.core.schema_validator import SchemaValidator

def test_finding_schema_validation():
    validator = SchemaValidator()
    
    # Test data
    finding_data = {
        "id": "VULN-001",
        "type": "vulnerability",
        "severity": 8,
        "title": "SQL Injection Vulnerability",
        "affected_resources": ["https://example.com/login"],
        "metadata": {"cve": "CVE-2024-1234"}
    }
    
    # Validate against generated schema
    result = validator.validate("Finding", finding_data)
    assert result.is_valid
    assert result.errors == []

def test_cross_language_compatibility():
    validator = SchemaValidator()
    
    # Test that same data validates in all languages
    test_data = {...}  # Sample data
    
    for lang in ["python", "typescript", "rust", "go"]:
        result = validator.validate_in_language("ScanReport", test_data, lang)
        assert result.is_valid, f"Failed validation in {lang}: {result.errors}"
```

## 🎨 Customization

### Custom Field Mappings
```python
# Configure custom type mappings
from plugins.aiva_converters.core.type_mapper import TypeMapper

mapper = TypeMapper()
mapper.add_mapping("python", "typescript", {
    "datetime": "Date",  # Use Date instead of string
    "Decimal": "number", # Map Decimal to number
    "UUID": "string"     # Map UUID to string
})
```

### Output Formatting
```python
# Configure output formatting
from plugins.aiva_converters.core.formatter import CodeFormatter

formatter = CodeFormatter()
formatter.configure("typescript", {
    "indent_size": 2,
    "quote_style": "single",
    "trailing_comma": True
})
```

---

**Example Updated**: November 2, 2025  
**Plugin Version**: 1.0.0