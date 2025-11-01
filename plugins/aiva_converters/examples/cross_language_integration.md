# Cross-Language Integration Examples

## 📋 Overview

This guide demonstrates how to integrate and validate code across multiple programming languages using the AIVA Converters Plugin.

## 🔗 Multi-Language Project Structure

```
project/
├── python/
│   ├── schemas/
│   │   └── security_models.py      # Pydantic models
│   ├── services/
│   │   └── scanner_service.py      # Python implementation
│   └── tests/
├── typescript/
│   ├── interfaces/
│   │   └── security-models.ts      # Generated TypeScript interfaces
│   ├── services/
│   │   └── scanner-client.ts       # TypeScript client
│   └── tests/
├── rust/
│   ├── src/
│   │   ├── models/
│   │   │   └── security_scan.rs    # Generated Rust structs
│   │   └── services/
│   │       └── scanner_engine.rs   # Rust implementation
│   └── tests/
└── shared/
    ├── schemas/
    │   └── openapi.yaml            # API specification
    └── contracts/
        └── security_scan.proto     # Protocol Buffers definition
```

## 🎯 Unified Schema Definition

### Source Schema (Python/Pydantic)
```python
# python/schemas/security_models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

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

class ScanTarget(BaseModel):
    url: str = Field(description="Target URL to scan")
    scan_types: List[str] = Field(description="Types of scans to perform")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VulnerabilityFinding(BaseModel):
    id: str = Field(description="Unique finding identifier")
    title: str = Field(description="Vulnerability title")
    level: VulnerabilityLevel = Field(description="Severity level")
    description: Optional[str] = None
    location: Optional[str] = None
    remediation: Optional[str] = None
    cwe_id: Optional[int] = None
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0)

class ScanResult(BaseModel):
    scan_id: str = Field(description="Unique scan identifier")
    target: ScanTarget = Field(description="Scan target information")
    status: ScanStatus = Field(description="Current scan status")
    start_time: datetime = Field(description="Scan start timestamp")
    end_time: Optional[datetime] = None
    findings: List[VulnerabilityFinding] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
```

## 🔄 Cross-Language Generation

### Generate All Language Bindings
```bash
# Generate TypeScript interfaces
python plugins/aiva_converters/core/schema_codegen_tool.py \
  --input python/schemas/security_models.py \
  --lang typescript \
  --output typescript/interfaces/

# Generate Rust structs  
python plugins/aiva_converters/core/schema_codegen_tool.py \
  --input python/schemas/security_models.py \
  --lang rust \
  --output rust/src/models/

# Generate Go structs
python plugins/aiva_converters/core/schema_codegen_tool.py \
  --input python/schemas/security_models.py \
  --lang go \
  --output go/models/

# Generate C# classes
python plugins/aiva_converters/core/schema_codegen_tool.py \
  --input python/schemas/security_models.py \
  --lang csharp \
  --output csharp/Models/
```

## 🔧 Language-Specific Implementations

### Python Service Implementation
```python
# python/services/scanner_service.py
from schemas.security_models import ScanResult, ScanTarget, ScanStatus
import asyncio
from datetime import datetime

class SecurityScannerService:
    def __init__(self):
        self.active_scans: Dict[str, ScanResult] = {}
    
    async def start_scan(self, target: ScanTarget) -> str:
        """Start a new security scan"""
        scan_id = f"scan_{datetime.now().timestamp()}"
        
        scan_result = ScanResult(
            scan_id=scan_id,
            target=target,
            status=ScanStatus.PENDING,
            start_time=datetime.now()
        )
        
        self.active_scans[scan_id] = scan_result
        
        # Start async scan
        asyncio.create_task(self._perform_scan(scan_id))
        
        return scan_id
    
    async def get_scan_result(self, scan_id: str) -> Optional[ScanResult]:
        """Get scan result by ID"""
        return self.active_scans.get(scan_id)
    
    async def _perform_scan(self, scan_id: str):
        """Perform the actual security scan"""
        scan = self.active_scans[scan_id]
        scan.status = ScanStatus.RUNNING
        
        try:
            # Simulate scan logic
            await asyncio.sleep(5)
            
            # Add findings
            scan.findings.extend([
                VulnerabilityFinding(
                    id="VULN-001",
                    title="SQL Injection detected",
                    level=VulnerabilityLevel.HIGH,
                    cwe_id=89,
                    cvss_score=8.5
                )
            ])
            
            scan.status = ScanStatus.COMPLETED
            scan.end_time = datetime.now()
            
        except Exception as e:
            scan.status = ScanStatus.FAILED
            scan.end_time = datetime.now()
```

### TypeScript Client Implementation
```typescript
// typescript/services/scanner-client.ts
import { ScanResult, ScanTarget, ScanStatus, VulnerabilityLevel } from '../interfaces/security-models';

export class SecurityScannerClient {
    private baseUrl: string;
    
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }
    
    async startScan(target: ScanTarget): Promise<string> {
        const response = await fetch(`${this.baseUrl}/scans`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(target)
        });
        
        if (!response.ok) {
            throw new Error(`Failed to start scan: ${response.statusText}`);
        }
        
        const result = await response.json();
        return result.scan_id;
    }
    
    async getScanResult(scanId: string): Promise<ScanResult | null> {
        const response = await fetch(`${this.baseUrl}/scans/${scanId}`);
        
        if (response.status === 404) {
            return null;
        }
        
        if (!response.ok) {
            throw new Error(`Failed to get scan result: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Validate the response matches our interface
        if (!this.isValidScanResult(data)) {
            throw new Error('Invalid scan result format received');
        }
        
        return data;
    }
    
    private isValidScanResult(data: any): data is ScanResult {
        return (
            typeof data.scan_id === 'string' &&
            typeof data.target === 'object' &&
            Object.values(ScanStatus).includes(data.status) &&
            typeof data.start_time === 'string' &&
            Array.isArray(data.findings)
        );
    }
    
    async waitForCompletion(scanId: string, maxWaitMs: number = 30000): Promise<ScanResult> {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitMs) {
            const result = await this.getScanResult(scanId);
            
            if (!result) {
                throw new Error('Scan not found');
            }
            
            if (result.status === ScanStatus.COMPLETED || result.status === ScanStatus.FAILED) {
                return result;
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        throw new Error('Scan did not complete within timeout period');
    }
}
```

### Rust High-Performance Scanner Engine
```rust
// rust/src/services/scanner_engine.rs
use crate::models::security_scan::{ScanResult, ScanTarget, ScanStatus, VulnerabilityFinding, VulnerabilityLevel};
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub struct ScannerEngine {
    active_scans: RwLock<HashMap<String, ScanResult>>,
}

impl ScannerEngine {
    pub fn new() -> Self {
        Self {
            active_scans: RwLock::new(HashMap::new()),
        }
    }
    
    pub async fn start_scan(&self, target: ScanTarget) -> Result<String, ScanError> {
        let scan_id = Uuid::new_v4().to_string();
        
        let scan_result = ScanResult {
            scan_id: scan_id.clone(),
            target,
            status: ScanStatus::Pending,
            start_time: Utc::now(),
            end_time: None,
            findings: Vec::new(),
            summary: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        {
            let mut scans = self.active_scans.write().await;
            scans.insert(scan_id.clone(), scan_result);
        }
        
        // Spawn scan task
        let engine = self.clone();
        let scan_id_clone = scan_id.clone();
        tokio::spawn(async move {
            if let Err(e) = engine.perform_scan(&scan_id_clone).await {
                eprintln!("Scan failed: {}", e);
            }
        });
        
        Ok(scan_id)
    }
    
    pub async fn get_scan_result(&self, scan_id: &str) -> Option<ScanResult> {
        let scans = self.active_scans.read().await;
        scans.get(scan_id).cloned()
    }
    
    async fn perform_scan(&self, scan_id: &str) -> Result<(), ScanError> {
        // Update status to running
        {
            let mut scans = self.active_scans.write().await;
            if let Some(scan) = scans.get_mut(scan_id) {
                scan.status = ScanStatus::Running;
            }
        }
        
        // Simulate high-performance scanning
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        // Generate findings
        let findings = vec![
            VulnerabilityFinding {
                id: "VULN-RUST-001".to_string(),
                title: "Buffer overflow detected".to_string(),
                level: VulnerabilityLevel::Critical,
                description: Some("Potential buffer overflow in input handling".to_string()),
                location: Some("main.c:142".to_string()),
                remediation: Some("Use safe string functions".to_string()),
                cwe_id: Some(120),
                cvss_score: Some(9.3),
            }
        ];
        
        // Update scan with results
        {
            let mut scans = self.active_scans.write().await;
            if let Some(scan) = scans.get_mut(scan_id) {
                scan.findings = findings;
                scan.status = ScanStatus::Completed;
                scan.end_time = Some(Utc::now());
                
                // Update summary
                scan.summary.insert("total".to_string(), scan.findings.len() as i32);
                scan.summary.insert("critical".to_string(), 1);
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("Scan not found: {0}")]
    NotFound(String),
    #[error("Invalid target: {0}")]
    InvalidTarget(String),
    #[error("Scan engine error: {0}")]
    EngineError(String),
}
```

## 🔄 Cross-Language Communication

### REST API Bridge
```python
# python/api/scanner_api.py
from fastapi import FastAPI, HTTPException
from schemas.security_models import ScanResult, ScanTarget
from services.scanner_service import SecurityScannerService

app = FastAPI(title="Security Scanner API", version="1.0.0")
scanner_service = SecurityScannerService()

@app.post("/scans", response_model=dict)
async def start_scan(target: ScanTarget):
    try:
        scan_id = await scanner_service.start_scan(target)
        return {"scan_id": scan_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scans/{scan_id}", response_model=ScanResult)
async def get_scan(scan_id: str):
    result = await scanner_service.get_scan_result(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scan not found")
    return result
```

### gRPC Integration
```protobuf
// shared/contracts/security_scan.proto
syntax = "proto3";

package security_scan;

service SecurityScanner {
  rpc StartScan(ScanTarget) returns (StartScanResponse);
  rpc GetScanResult(GetScanRequest) returns (ScanResult);
  rpc StreamScanUpdates(GetScanRequest) returns (stream ScanResult);
}

message ScanTarget {
  string url = 1;
  repeated string scan_types = 2;
  map<string, string> metadata = 3;
}

message ScanResult {
  string scan_id = 1;
  ScanTarget target = 2;
  ScanStatus status = 3;
  int64 start_time = 4;
  optional int64 end_time = 5;
  repeated VulnerabilityFinding findings = 6;
  map<string, int32> summary = 7;
}

enum ScanStatus {
  PENDING = 0;
  RUNNING = 1;
  COMPLETED = 2;
  FAILED = 3;
}

message VulnerabilityFinding {
  string id = 1;
  string title = 2;
  VulnerabilityLevel level = 3;
  optional string description = 4;
  optional string location = 5;
  optional string remediation = 6;
  optional int32 cwe_id = 7;
  optional double cvss_score = 8;
}

enum VulnerabilityLevel {
  INFO = 0;
  LOW = 1;
  MEDIUM = 2;
  HIGH = 3;
  CRITICAL = 4;
}
```

## 🧪 Cross-Language Testing

### Integration Test Suite
```python
# tests/integration/test_cross_language.py
import pytest
import asyncio
import subprocess
import time
from python.services.scanner_service import SecurityScannerService
from python.schemas.security_models import ScanTarget

class TestCrossLanguageIntegration:
    
    @pytest.fixture(scope="class")
    async def services(self):
        # Start Python service
        python_service = SecurityScannerService()
        
        # Start TypeScript service (via subprocess)
        ts_process = subprocess.Popen([
            "npm", "start"
        ], cwd="typescript/")
        
        # Start Rust service
        rust_process = subprocess.Popen([
            "cargo", "run", "--release"
        ], cwd="rust/")
        
        # Wait for services to start
        time.sleep(5)
        
        yield {
            "python": python_service,
            "typescript": ts_process,
            "rust": rust_process
        }
        
        # Cleanup
        ts_process.terminate()
        rust_process.terminate()
    
    async def test_python_to_typescript_compatibility(self, services):
        """Test that Python service output is compatible with TypeScript client"""
        
        target = ScanTarget(
            url="https://example.com",
            scan_types=["vulnerability", "malware"]
        )
        
        # Start scan with Python service
        scan_id = await services["python"].start_scan(target)
        
        # Wait for completion
        result = None
        for _ in range(30):  # Wait up to 30 seconds
            result = await services["python"].get_scan_result(scan_id)
            if result and result.status.value in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
        
        assert result is not None
        
        # Verify TypeScript client can parse the result
        # This would involve HTTP requests to TypeScript service
        # and validating the response structure
    
    def test_schema_compatibility_across_languages(self):
        """Test that generated schemas are compatible across all languages"""
        
        from plugins.aiva_converters.core.cross_language_validator import CrossLanguageValidator
        
        validator = CrossLanguageValidator()
        
        # Test core types compatibility
        compatibility_results = validator.validate_multi_language_schemas(
            schema_name="ScanResult",
            languages=["python", "typescript", "rust", "go"]
        )
        
        for language, result in compatibility_results.items():
            assert result.is_compatible, f"Schema incompatibility in {language}: {result.issues}"
            assert result.compatibility_score >= 95, f"Low compatibility score for {language}: {result.compatibility_score}%"
    
    def test_data_serialization_roundtrip(self):
        """Test data can roundtrip through all language implementations"""
        
        # Create test data in Python format
        original_data = {
            "scan_id": "test-123",
            "target": {
                "url": "https://test.com",
                "scan_types": ["vulnerability"],
                "metadata": {}
            },
            "status": "completed",
            "start_time": "2024-11-02T10:30:00Z",
            "findings": [],
            "summary": {}
        }
        
        # Test serialization/deserialization through each language
        languages = ["python", "typescript", "rust", "go"]
        
        current_data = original_data
        for lang in languages:
            # This would involve HTTP calls to each service
            # to serialize and deserialize the data
            serialized = self._serialize_in_language(current_data, lang)
            deserialized = self._deserialize_in_language(serialized, lang)
            current_data = deserialized
        
        # Verify data integrity after full roundtrip
        assert self._normalize_data(current_data) == self._normalize_data(original_data)
```

### Performance Benchmarking
```python
# tests/performance/benchmark_cross_language.py
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class CrossLanguagePerformanceBenchmark:
    
    async def benchmark_scan_performance(self):
        """Benchmark scan performance across different language implementations"""
        
        target = ScanTarget(
            url="https://benchmark.example.com",
            scan_types=["vulnerability", "malware", "network"]
        )
        
        results = {}
        
        # Benchmark each language implementation
        languages = {
            "python": self._benchmark_python_scan,
            "rust": self._benchmark_rust_scan,
            "go": self._benchmark_go_scan
        }
        
        for lang_name, benchmark_func in languages.items():
            times = []
            
            # Run multiple iterations
            for _ in range(10):
                start_time = time.perf_counter()
                await benchmark_func(target)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[lang_name] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times)
            }
        
        # Print results
        print("\nCross-Language Performance Benchmark Results:")
        print("-" * 50)
        for lang, stats in results.items():
            print(f"{lang.upper()}:")
            print(f"  Mean: {stats['mean']:.3f}s")
            print(f"  Median: {stats['median']:.3f}s") 
            print(f"  Std Dev: {stats['std_dev']:.3f}s")
            print(f"  Range: {stats['min']:.3f}s - {stats['max']:.3f}s")
            print()
        
        return results
```

## 📊 Monitoring and Observability

### Unified Telemetry
```python
# shared/telemetry/cross_language_metrics.py
from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class CrossLanguageMetric:
    service_name: str
    language: str
    operation: str
    duration_ms: float
    success: bool
    metadata: Dict[str, Any]

class TelemetryCollector:
    def __init__(self):
        self.metrics: List[CrossLanguageMetric] = []
    
    def record_operation(self, metric: CrossLanguageMetric):
        self.metrics.append(metric)
        
        # Send to monitoring system
        self._send_to_monitoring(metric)
    
    def _send_to_monitoring(self, metric: CrossLanguageMetric):
        # Send to Prometheus, DataDog, etc.
        metric_data = {
            "service": metric.service_name,
            "language": metric.language,
            "operation": metric.operation,
            "duration": metric.duration_ms,
            "success": metric.success,
            **metric.metadata
        }
        
        # Example: Send to monitoring endpoint
        # requests.post("http://metrics-collector/metrics", json=metric_data)
```

---

**Example Updated**: November 2, 2025  
**Plugin Version**: 1.0.0