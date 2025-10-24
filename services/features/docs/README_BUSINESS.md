# AIVA Features - 業務功能架構 🏢

> **定位**: 功能實現層、服務提供、業務邏輯處理  
> **規模**: 174 個業務組件 (6.5%)  
> **職責**: SCA、CSPM、認證服務、注入檢測、SSRF 防護

---

## 🎯 **業務功能在 AIVA 中的角色**

### **🚀 功能實現定位**
業務功能層是 AIVA Features 的「**功能實現層**」，將安全能力轉化為具體的業務服務：

```
🏢 業務功能實現架構
├── 🔍 軟體組件分析 (SCA) - 20組件
│   ├── 依賴項掃描 (Go 高效能)
│   ├── 漏洞資料庫比對
│   └── License 合規檢查
├── ☁️ 雲端安全態勢管理 (CSPM) - 15組件
│   ├── AWS 配置檢查 (Go 並發)
│   ├── Azure 安全評估
│   └── 多雲合規掃描
├── 🔐 認證安全服務 (AUTHN) - 15組件
│   ├── JWT 令牌驗證 (Go 效能)
│   ├── OAuth 2.0 流程
│   └── 多因素認證
├── 💉 SQL 注入檢測服務 - 19組件
│   ├── 動態 SQL 分析 (Python 靈活)
│   ├── 語法樹解析
│   └── 注入模式識別  
└── 🌐 SSRF 檢測防護服務 - 19組件
    ├── URL 驗證 (Go 並發)
    ├── 內網訪問控制
    └── DNS 重綁定防護
```

### **⚡ 業務組件統計**
- **SCA 軟體組件分析**: 20 個組件 (11.5% - 供應鏈安全)
- **SQL 注入檢測**: 19 個組件 (10.9% - Web 安全核心)
- **SSRF 檢測防護**: 19 個組件 (10.9% - 網路安全)
- **CSPM 雲端安全**: 15 個組件 (8.6% - 雲原生安全)
- **認證安全服務**: 15 個組件 (8.6% - 身份驗證)

---

## 🚨 **發現的架構問題與改進建議**

### **⚠️ 重複模組問題**

#### **問題 1: SQL 注入檢測重複**
```
❌ 當前狀況:
- Core Layer: SQL_Injection_Detection (7 組件)
- Security Layer: SQL_Injection_Detection (59 組件)  
- Business Layer: SQL_Injection_Detection (19 組件)
- Support Layer: SQL_Injection_Detection (1 組件)

✅ 建議改進:
- Core Layer: 保留智能協調邏輯
- Security Layer: 保留核心檢測引擎 (Rust)
- Business Layer: 提供 REST API 服務 (Python)
- Support Layer: 僅保留配置 Schema
```

#### **問題 2: SSRF 檢測重複**
```
❌ 當前狀況:
- Security Layer: SSRF_Detection (58 組件)
- Business Layer: SSRF_Detection (19 組件)
- Support Layer: SSRF_Detection (2 組件)

✅ 建議改進:
- Security Layer: Rust 核心檢測引擎
- Business Layer: Go 高效能服務 API
- Support Layer: Python 配置管理
```

#### **問題 3: 認證服務語言不一致**
```
❌ 當前狀況:
- Business Layer: Authentication_Security (15組件 Go)
- Support Layer: Authentication_Security (29組件 Python)

✅ 建議改進:
- Business Layer: Go 高效能認證服務
- Support Layer: Python 認證配置與策略管理
```

### **🔧 架構重構建議**

#### **建議的分層職責重新定義**
```python
"""
推薦的業務功能分層架構
"""

class BusinessLayerArchitecture:
    """業務功能層架構重構"""
    
    def __init__(self):
        self.layers = {
            "api_service": {
                "language": "Python",
                "components": ["REST API", "GraphQL", "WebSocket"],
                "responsibility": "對外服務介面"
            },
            "business_logic": {
                "language": "Python + Go", 
                "components": ["SCA Service", "CSPM Service", "Auth Service"],
                "responsibility": "業務邏輯實現"
            },
            "security_integration": {
                "language": "Rust + Python",
                "components": ["Security Engine Wrapper", "Result Processor"],
                "responsibility": "安全引擎整合"
            }
        }
    
    def get_recommended_structure(self):
        return {
            "sca_service": {
                "api_layer": "Python FastAPI",
                "business_logic": "Go 並發處理", 
                "security_engine": "Rust 核心掃描",
                "data_layer": "Python ORM"
            },
            "cspm_service": {
                "api_layer": "Python FastAPI",
                "business_logic": "Go 雲端 API 整合",
                "security_engine": "Rust 規則引擎", 
                "data_layer": "Python 配置管理"
            }
        }
```

---

## 🏗️ **業務功能架構模式**

### **🔍 SCA 軟體組件分析服務**

```python
"""
SCA (Software Composition Analysis) 業務服務
整合 Rust 核心引擎與 Go 高效能處理
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import aioredis

class ScanStatus(Enum):
    """掃描狀態"""
    PENDING = "pending"
    SCANNING = "scanning" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SCARequest:
    """SCA 掃描請求"""
    project_id: str
    repository_url: str
    branch: str = "main"
    scan_depth: str = "deep"
    include_dev_dependencies: bool = True
    exclude_patterns: List[str] = None

@dataclass
class Vulnerability:
    """漏洞資訊"""
    cve_id: str
    severity: str
    cvss_score: float
    affected_package: str
    affected_version: str
    fixed_version: Optional[str]
    description: str
    references: List[str]

@dataclass
class SCAResult:
    """SCA 掃描結果"""
    scan_id: str
    project_id: str
    status: ScanStatus
    total_packages: int
    vulnerable_packages: int
    vulnerabilities: List[Vulnerability]
    license_issues: List[Dict[str, Any]]
    risk_score: float
    scan_duration: float
    timestamp: str

class SCABusinessService:
    """SCA 業務服務"""
    
    def __init__(self):
        self.rust_engine_client = RustEngineClient()
        self.go_processing_client = GoProcessingClient()
        self.redis_client = None
        self.vulnerability_db = VulnerabilityDatabase()
        
    async def initialize(self):
        """初始化服務"""
        self.redis_client = await aioredis.from_url("redis://localhost")
        await self.vulnerability_db.initialize()
        
    async def start_scan(self, request: SCARequest) -> str:
        """啟動 SCA 掃描"""
        # 1. 生成掃描 ID
        scan_id = f"sca_{request.project_id}_{int(time.time())}"
        
        # 2. 初始化掃描狀態
        scan_result = SCAResult(
            scan_id=scan_id,
            project_id=request.project_id,
            status=ScanStatus.PENDING,
            total_packages=0,
            vulnerable_packages=0,
            vulnerabilities=[],
            license_issues=[],
            risk_score=0.0,
            scan_duration=0.0,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # 3. 存儲到 Redis
        await self.redis_client.set(
            f"scan:{scan_id}",
            json.dumps(asdict(scan_result)),
            ex=86400  # 24小時過期
        )
        
        # 4. 啟動背景掃描任務
        asyncio.create_task(self._perform_scan(scan_id, request))
        
        return scan_id
    
    async def _perform_scan(self, scan_id: str, request: SCARequest):
        """執行 SCA 掃描"""
        try:
            # 1. 更新狀態為掃描中
            await self._update_scan_status(scan_id, ScanStatus.SCANNING)
            
            # 2. 調用 Go 服務處理代碼庫
            repository_data = await self.go_processing_client.process_repository(
                url=request.repository_url,
                branch=request.branch,
                depth=request.scan_depth
            )
            
            # 3. 調用 Rust 引擎進行依賴分析
            dependencies = await self.rust_engine_client.analyze_dependencies(
                repository_data=repository_data,
                include_dev=request.include_dev_dependencies,
                exclude_patterns=request.exclude_patterns or []
            )
            
            # 4. 漏洞檢查
            vulnerabilities = await self._check_vulnerabilities(dependencies)
            
            # 5. License 檢查  
            license_issues = await self._check_licenses(dependencies)
            
            # 6. 計算風險分數
            risk_score = await self._calculate_risk_score(vulnerabilities)
            
            # 7. 組裝結果
            result = SCAResult(
                scan_id=scan_id,
                project_id=request.project_id,
                status=ScanStatus.COMPLETED,
                total_packages=len(dependencies),
                vulnerable_packages=len([d for d in dependencies if d.has_vulnerabilities]),
                vulnerabilities=vulnerabilities,
                license_issues=license_issues,
                risk_score=risk_score,
                scan_duration=(time.time() - start_time),
                timestamp=datetime.utcnow().isoformat()
            )
            
            # 8. 更新結果
            await self.redis_client.set(
                f"scan:{scan_id}",
                json.dumps(asdict(result)),
                ex=86400
            )
            
        except Exception as e:
            # 掃描失敗
            await self._update_scan_status(scan_id, ScanStatus.FAILED)
            logger.error(f"SCA scan failed: {scan_id}, error: {str(e)}")
    
    async def get_scan_result(self, scan_id: str) -> SCAResult:
        """獲取掃描結果"""
        data = await self.redis_client.get(f"scan:{scan_id}")
        if not data:
            raise HTTPException(status_code=404, detail="Scan not found")
            
        result_dict = json.loads(data)
        return SCAResult(**result_dict)
    
    async def _check_vulnerabilities(self, dependencies: List[Dependency]) -> List[Vulnerability]:
        """檢查漏洞"""
        vulnerabilities = []
        
        for dep in dependencies:
            # 查詢漏洞資料庫
            vulns = await self.vulnerability_db.get_vulnerabilities(
                package_name=dep.name,
                version=dep.version,
                ecosystem=dep.ecosystem
            )
            vulnerabilities.extend(vulns)
            
        return vulnerabilities
    
    async def _calculate_risk_score(self, vulnerabilities: List[Vulnerability]) -> float:
        """計算風險分數"""
        if not vulnerabilities:
            return 0.0
            
        total_score = 0.0
        for vuln in vulnerabilities:
            # 基於 CVSS 分數和嚴重程度計算
            if vuln.severity == "CRITICAL":
                total_score += vuln.cvss_score * 2.0
            elif vuln.severity == "HIGH":
                total_score += vuln.cvss_score * 1.5
            elif vuln.severity == "MEDIUM":
                total_score += vuln.cvss_score * 1.0
            else:
                total_score += vuln.cvss_score * 0.5
                
        # 正規化到 0-100
        return min(100.0, total_score / len(vulnerabilities))

class RustEngineClient:
    """Rust 引擎客戶端"""
    
    async def analyze_dependencies(
        self,
        repository_data: Dict[str, Any],
        include_dev: bool = True,
        exclude_patterns: List[str] = None
    ) -> List[Dict[str, Any]]:
        """調用 Rust 引擎分析依賴項"""
        # 這裡應該調用 Rust 微服務
        # 暫時返回模擬資料
        return [
            {
                "name": "requests",
                "version": "2.25.1",
                "ecosystem": "pypi",
                "has_vulnerabilities": True,
                "license": "Apache-2.0"
            }
        ]

class GoProcessingClient:
    """Go 處理服務客戶端"""
    
    async def process_repository(
        self,
        url: str,
        branch: str,
        depth: str
    ) -> Dict[str, Any]:
        """調用 Go 服務處理代碼庫"""
        # 這裡應該調用 Go 微服務
        return {
            "repository_url": url,
            "branch": branch,
            "files_processed": 150,
            "manifest_files": ["requirements.txt", "package.json", "go.mod"]
        }
```

### **☁️ CSPM 雲端安全態勢管理**

```go
// CSPM (Cloud Security Posture Management) 業務服務
// Go 實現高效能雲端資源掃描與合規檢查

package cspm

import (
    "context"
    "encoding/json"
    "fmt"
    "sync"
    "time"
    
    "github.com/aws/aws-sdk-go-v2/aws"
    "github.com/aws/aws-sdk-go-v2/service/ec2"
    "github.com/aws/aws-sdk-go-v2/service/s3"
    "go.uber.org/zap"
)

// CSPMBusinessService CSPM 業務服務
type CSPMBusinessService struct {
    logger          *zap.Logger
    awsConfig       aws.Config
    complianceRules []ComplianceRule
    resultStore     ResultStore
}

// ScanRequest CSMP 掃描請求
type ScanRequest struct {
    AccountID      string   `json:"account_id"`
    Regions        []string `json:"regions"`
    ResourceTypes  []string `json:"resource_types"`
    ComplianceFramework string `json:"compliance_framework"` // SOC2, PCI-DSS, ISO27001
    IncludeRemediation bool  `json:"include_remediation"`
}

// ComplianceResult 合規檢查結果
type ComplianceResult struct {
    ScanID          string                 `json:"scan_id"`
    AccountID       string                 `json:"account_id"`
    Framework       string                 `json:"framework"`
    OverallScore    float64               `json:"overall_score"`
    TotalChecks     int                   `json:"total_checks"`
    PassedChecks    int                   `json:"passed_checks"`
    FailedChecks    int                   `json:"failed_checks"`
    Findings        []ComplianceFinding   `json:"findings"`
    ScanDuration    time.Duration         `json:"scan_duration"`
    Timestamp       time.Time             `json:"timestamp"`
}

// ComplianceFinding 合規發現
type ComplianceFinding struct {
    RuleID          string                 `json:"rule_id"`
    RuleName        string                 `json:"rule_name"`
    Severity        string                 `json:"severity"`
    Status          string                 `json:"status"` // PASS, FAIL, MANUAL
    ResourceID      string                 `json:"resource_id"`
    ResourceType    string                 `json:"resource_type"`
    Region          string                 `json:"region"`
    Description     string                 `json:"description"`
    Remediation     *RemediationAdvice    `json:"remediation,omitempty"`
    Evidence        map[string]interface{} `json:"evidence"`
}

// RemediationAdvice 修復建議
type RemediationAdvice struct {
    Action          string            `json:"action"`
    Instructions    string            `json:"instructions"`
    AWSCLICommand   string            `json:"aws_cli_command,omitempty"`
    TerraformCode   string            `json:"terraform_code,omitempty"`
    RiskLevel       string            `json:"risk_level"`
    EstimatedEffort string            `json:"estimated_effort"`
}

// NewCSPMBusinessService 建立 CSPM 業務服務
func NewCSPMBusinessService(awsConfig aws.Config, logger *zap.Logger) *CSPMBusinessService {
    service := &CSPMBusinessService{
        logger:    logger,
        awsConfig: awsConfig,
        resultStore: NewRedisResultStore(),
    }
    
    // 載入合規規則
    service.loadComplianceRules()
    
    return service
}

// StartScan 開始 CSPM 掃描
func (c *CSPMBusinessService) StartScan(ctx context.Context, req *ScanRequest) (string, error) {
    scanID := generateScanID(req.AccountID)
    
    c.logger.Info("Starting CSPM scan",
        zap.String("scan_id", scanID),
        zap.String("account_id", req.AccountID),
        zap.Strings("regions", req.Regions),
    )
    
    // 異步執行掃描
    go c.performScan(ctx, scanID, req)
    
    return scanID, nil
}

// performScan 執行 CSPM 掃描
func (c *CSPMBusinessService) performScan(ctx context.Context, scanID string, req *ScanRequest) {
    startTime := time.Now()
    
    defer func() {
        if r := recover(); r != nil {
            c.logger.Error("CSPM scan panicked",
                zap.String("scan_id", scanID),
                zap.Any("panic", r),
            )
        }
    }()
    
    // 1. 收集雲端資源
    resources, err := c.collectCloudResources(ctx, req)
    if err != nil {
        c.logger.Error("Failed to collect cloud resources",
            zap.String("scan_id", scanID),
            zap.Error(err),
        )
        return
    }
    
    // 2. 並行執行合規檢查
    findings := c.runComplianceChecks(ctx, resources, req.ComplianceFramework)
    
    // 3. 計算合規分數
    score := c.calculateComplianceScore(findings)
    
    // 4. 生成修復建議
    if req.IncludeRemediation {
        c.generateRemediationAdvice(findings)
    }
    
    // 5. 組裝結果
    result := &ComplianceResult{
        ScanID:       scanID,
        AccountID:    req.AccountID,
        Framework:    req.ComplianceFramework,
        OverallScore: score,
        TotalChecks:  len(c.complianceRules),
        PassedChecks: countPassedChecks(findings),
        FailedChecks: countFailedChecks(findings),
        Findings:     findings,
        ScanDuration: time.Since(startTime),
        Timestamp:    time.Now(),
    }
    
    // 6. 存儲結果
    if err := c.resultStore.Store(scanID, result); err != nil {
        c.logger.Error("Failed to store scan result",
            zap.String("scan_id", scanID),
            zap.Error(err),
        )
    }
    
    c.logger.Info("CSMP scan completed",
        zap.String("scan_id", scanID),
        zap.Float64("score", score),
        zap.Duration("duration", result.ScanDuration),
    )
}

// runComplianceChecks 並行執行合規檢查
func (c *CSPMBusinessService) runComplianceChecks(
    ctx context.Context,
    resources []CloudResource,
    framework string,
) []ComplianceFinding {
    
    // 篩選適用的規則
    applicableRules := c.getApplicableRules(framework)
    
    // 建立結果通道
    findingsChan := make(chan ComplianceFinding, len(applicableRules)*len(resources))
    
    // 並發檢查
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, 10) // 限制並發數
    
    for _, rule := range applicableRules {
        for _, resource := range resources {
            // 檢查規則是否適用於資源類型
            if !rule.AppliesToResourceType(resource.Type()) {
                continue
            }
            
            wg.Add(1)
            go func(r ComplianceRule, res CloudResource) {
                defer wg.Done()
                
                // 獲取信號量
                semaphore <- struct{}{}
                defer func() { <-semaphore }()
                
                // 執行規則檢查
                finding, err := r.Check(ctx, res)
                if err != nil {
                    c.logger.Warn("Rule check failed",
                        zap.String("rule_id", r.ID()),
                        zap.String("resource_id", res.ID()),
                        zap.Error(err),
                    )
                    return
                }
                
                if finding != nil {
                    findingsChan <- *finding
                }
            }(rule, resource)
        }
    }
    
    // 等待所有檢查完成
    go func() {
        wg.Wait()
        close(findingsChan)
    }()
    
    // 收集結果
    var findings []ComplianceFinding
    for finding := range findingsChan {
        findings = append(findings, finding)
    }
    
    return findings
}

// calculateComplianceScore 計算合規分數
func (c *CSPMBusinessService) calculateComplianceScore(findings []ComplianceFinding) float64 {
    if len(findings) == 0 {
        return 100.0
    }
    
    totalWeight := 0.0
    passedWeight := 0.0
    
    for _, finding := range findings {
        weight := c.getRuleWeight(finding.RuleID, finding.Severity)
        totalWeight += weight
        
        if finding.Status == "PASS" {
            passedWeight += weight
        }
    }
    
    if totalWeight == 0 {
        return 100.0
    }
    
    return (passedWeight / totalWeight) * 100.0
}

// getRuleWeight 獲取規則權重
func (c *CSPMBusinessService) getRuleWeight(ruleID, severity string) float64 {
    switch severity {
    case "CRITICAL":
        return 10.0
    case "HIGH":
        return 7.0
    case "MEDIUM":
        return 4.0
    case "LOW":
        return 1.0
    default:
        return 1.0
    }
}

// 特定的 AWS 安全檢查規則
type EC2SecurityGroupRule struct {
    ID   string
    Name string
}

// Check 實現 EC2 安全組檢查
func (r *EC2SecurityGroupRule) Check(ctx context.Context, resource CloudResource) (*ComplianceFinding, error) {
    sg, ok := resource.(*EC2SecurityGroup)
    if !ok {
        return nil, fmt.Errorf("invalid resource type for EC2SecurityGroupRule")
    }
    
    // 檢查是否允許來自 0.0.0.0/0 的入站流量
    hasOpenAccess := false
    for _, rule := range sg.InboundRules {
        if rule.Source == "0.0.0.0/0" && (rule.Port == 22 || rule.Port == 3389) {
            hasOpenAccess = true
            break
        }
    }
    
    status := "PASS"
    description := "Security group properly configured"
    
    if hasOpenAccess {
        status = "FAIL"
        description = "Security group allows unrestricted access to sensitive ports"
    }
    
    return &ComplianceFinding{
        RuleID:       r.ID,
        RuleName:     r.Name,
        Severity:     "HIGH",
        Status:       status,
        ResourceID:   sg.ID(),
        ResourceType: "AWS::EC2::SecurityGroup",
        Region:       sg.Region(),
        Description:  description,
        Evidence: map[string]interface{}{
            "open_rules": sg.getOpenRules(),
        },
    }, nil
}
```

---

## 🔧 **業務層整合架構**

### **🌐 統一 API 閘道**
```python
"""
業務功能統一 API 閘道
整合所有業務服務的對外介面
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
from typing import Dict, Any

app = FastAPI(
    title="AIVA Business Functions API",
    description="AIVA 業務功能統一服務 API",
    version="2.0.0"
)

# 業務服務實例
sca_service = None
csmp_service = None
auth_service = None
injection_service = None
ssrf_service = None

@app.on_event("startup")
async def startup_event():
    """應用啟動事件"""
    global sca_service, csmp_service, auth_service, injection_service, ssrf_service
    
    # 初始化各業務服務
    sca_service = SCABusinessService()
    await sca_service.initialize()
    
    csmp_service = CSPMBusinessService()
    await csmp_service.initialize()
    
    # 其他服務初始化...

# SCA API 路由
@app.post("/api/v1/sca/scan")
async def start_sca_scan(request: SCARequest):
    """啟動 SCA 掃描"""
    scan_id = await sca_service.start_scan(request)
    return {"scan_id": scan_id, "status": "started"}

@app.get("/api/v1/sca/scan/{scan_id}")
async def get_sca_result(scan_id: str):
    """獲取 SCA 掃描結果"""
    result = await sca_service.get_scan_result(scan_id)
    return result

# CSPM API 路由
@app.post("/api/v1/cspm/scan") 
async def start_cspm_scan(request: Dict[str, Any]):
    """啟動 CSPM 掃描"""
    # 調用 Go 服務
    scan_id = await cspm_service.start_scan(request)
    return {"scan_id": scan_id, "status": "started"}

# 健康檢查
@app.get("/health")
async def health_check():
    """服務健康檢查"""
    services_status = {
        "sca_service": await check_service_health(sca_service),
        "cspm_service": await check_service_health(csmp_service),
        "overall": "healthy"
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services_status
    }
```

---

**📝 版本**: v2.0 - Business Functions Architecture Guide  
**🔄 最後更新**: 2024-10-24  
**🏢 主要語言**: Python (API) + Go (處理) + Rust (引擎)  
**👥 維護團隊**: AIVA Business Layer Team

**⚠️ 架構改進優先級**:
1. **立即處理**: 解決模組重複問題
2. **短期**: 重新定義分層職責  
3. **中期**: 統一跨語言介面標準
4. **長期**: 實現完全的微服務化架構

*這是 AIVA Features 模組業務功能組件的完整架構指南，包含發現的架構問題分析和改進建議。*