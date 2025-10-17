# AIVA 跨模組流程與 CLI 命令對應

生成時間: 2025-10-17 10:49:21

## 📊 流程總覽

| 流程名稱 | 涉及模組 | CLI 命令數 | Topic 數 |
|---------|---------|-----------|---------|
| SSRF 檢測流程 | Core, Function, MQ | 1 | 2 |
| SQL 注入檢測流程 | Core, Function, MQ | 1 | 4 |
| 修復建議流程 | Core, Remediation, MQ | 2 | 3 |
| 掃描流程 | Core, Scan, MQ | 3 | 7 |
| IDOR 檢測流程 | Core, Function, MQ | 1 | 2 |
| 權限檢測流程 | Core, AuthZ, MQ | 2 | 3 |
| 整合分析流程 | Core, Integration, MQ | 3 | 7 |
| XSS 檢測流程 | Core, Function, MQ | 1 | 2 |
| AI 訓練流程 | Core, MQ | 4 | 10 |
| 威脅情報流程 | Core, ThreatIntel, MQ | 3 | 5 |
## 📋 詳細命令列表

### SSRF 檢測流程

**描述**: Core → Function(SSRF) → MQ → Core  
**涉及模組**: Core → Function → MQ

#### 命令

##### `aiva detect ssrf <url>`

**選項**:
- `--param <name>`
- `--callback-url <url>`
- `--wait`
- `--format json`

**對應 Topics**:
- `TASK_FUNCTION_SSRF`
- `RESULTS_FUNCTION_COMPLETED`

### SQL 注入檢測流程

**描述**: Core → Function(SQLi) → MQ → Core  
**涉及模組**: Core → Function → MQ

#### 命令

##### `aiva detect sqli <url>`

**選項**:
- `--param <name>`
- `--method <GET|POST>`
- `--engines <list>`
- `--wait`
- `--format json`

**對應 Topics**:
- `TASK_FUNCTION_SQLI`
- `RESULTS_FUNCTION_PROGRESS`
- `EVENT_FUNCTION_VULN_FOUND`
- `RESULTS_FUNCTION_COMPLETED`

### 修復建議流程

**描述**: Core → Remediation → MQ → Core  
**涉及模組**: Core → Remediation → MQ

#### 命令

##### `aiva remedy generate <vuln_id>`

**選項**:
- `--language <python|java|go|...>`
- `--framework <flask|django|...>`
- `--format json`

**對應 Topics**:
- `TASK_REMEDIATION_GENERATE`
- `RESULTS_REMEDIATION`

##### `aiva remedy batch <scan_id>`

**選項**:
- `--format json`

**對應 Topics**:
- `TASK_REMEDIATION_BATCH`
- `RESULTS_REMEDIATION`

### 掃描流程

**描述**: User → API → Core → Scan → MQ  
**涉及模組**: Core → Scan → MQ

#### 命令

##### `aiva scan start <url>`

**選項**:
- `--max-depth <int>`
- `--max-pages <int>`
- `--wait`
- `--format json`

**對應 Topics**:
- `TASK_SCAN_START`
- `RESULTS_SCAN_PROGRESS`
- `RESULTS_SCAN_COMPLETED`

##### `aiva scan status <scan_id>`

**選項**:
- `--format json`

**對應 Topics**:
- `QUERY_SCAN_STATUS`
- `RESULTS_SCAN_STATUS`

##### `aiva scan assets <scan_id>`

**選項**:
- `--type <web|api|file>`
- `--format json`

**對應 Topics**:
- `QUERY_SCAN_ASSETS`
- `RESULTS_SCAN_ASSETS`

### IDOR 檢測流程

**描述**: Core → Function(IDOR) → MQ → Core  
**涉及模組**: Core → Function → MQ

#### 命令

##### `aiva detect idor <url>`

**選項**:
- `--param <name>`
- `--user-context <json>`
- `--wait`
- `--format json`

**對應 Topics**:
- `FUNCTION_IDOR_TASK`
- `RESULTS_FUNCTION_COMPLETED`

### 權限檢測流程

**描述**: Core → AuthZ → MQ → Core  
**涉及模組**: Core → AuthZ → MQ

#### 命令

##### `aiva authz check <url>`

**選項**:
- `--user-context <json>`
- `--test-escalation`
- `--format json`

**對應 Topics**:
- `TASK_AUTHZ_CHECK`
- `RESULTS_AUTHZ`

##### `aiva authz analyze <scan_id>`

**選項**:
- `--format json`

**對應 Topics**:
- `TASK_AUTHZ_ANALYZE`
- `RESULTS_AUTHZ`

### 整合分析流程

**描述**: Core → Integration → MQ → Core  
**涉及模組**: Core → Integration → MQ

#### 命令

##### `aiva report generate <scan_id>`

**選項**:
- `--format <pdf|html|json>`
- `--output <file>`
- `--no-findings`
- `--include-remediation`

**對應 Topics**:
- `TASK_INTEGRATION_ANALYSIS_START`
- `RESULTS_INTEGRATION_ANALYSIS_COMPLETED`
- `COMMAND_INTEGRATION_REPORT_GENERATE`
- `EVENT_INTEGRATION_REPORT_GENERATED`

##### `aiva report status <analysis_id>`

**選項**:
- `--format json`

**對應 Topics**:
- `QUERY_INTEGRATION_STATUS`
- `RESULTS_INTEGRATION_STATUS`

##### `aiva report export <scan_id>`

**選項**:
- `--format <csv|excel|json>`
- `--output <file>`

**對應 Topics**:
- `COMMAND_INTEGRATION_EXPORT`

### XSS 檢測流程

**描述**: Core → Function(XSS) → MQ → Core  
**涉及模組**: Core → Function → MQ

#### 命令

##### `aiva detect xss <url>`

**選項**:
- `--param <name>`
- `--type <reflected|stored|dom>`
- `--wait`
- `--format json`

**對應 Topics**:
- `TASK_FUNCTION_XSS`
- `RESULTS_FUNCTION_COMPLETED`

### AI 訓練流程

**描述**: Core(AI) → MQ → Knowledge Base  
**涉及模組**: Core → MQ

#### 命令

##### `aiva ai train `

**選項**:
- `--mode <realtime|replay|simulation>`
- `--epochs <int>`
- `--scenarios <int>`
- `--storage-path <path>`
- `--format json`

**對應 Topics**:
- `TASK_AI_TRAINING_START`
- `TASK_AI_TRAINING_EPISODE`
- `RESULTS_AI_TRAINING_PROGRESS`
- `EVENT_AI_EXPERIENCE_CREATED`
- `EVENT_AI_MODEL_UPDATED`
- `RESULTS_AI_TRAINING_COMPLETED`

##### `aiva ai status `

**選項**:
- `--storage-path <path>`
- `--format json`

**對應 Topics**:
- `QUERY_AI_STATUS`
- `RESULTS_AI_STATUS`

##### `aiva ai stop <training_id>`

**選項**:
- `--format json`

**對應 Topics**:
- `TASK_AI_TRAINING_STOP`

##### `aiva ai deploy <model_id>`

**選項**:
- `--environment <dev|staging|prod>`
- `--format json`

**對應 Topics**:
- `COMMAND_AI_MODEL_DEPLOY`

### 威脅情報流程

**描述**: Core → Threat Intel → External APIs → Core  
**涉及模組**: Core → ThreatIntel → MQ

#### 命令

##### `aiva threat lookup <ioc>`

**選項**:
- `--type <ip|domain|hash|url>`
- `--enrich`
- `--mitre`
- `--format json`

**對應 Topics**:
- `TASK_THREAT_INTEL_LOOKUP`
- `TASK_IOC_ENRICHMENT`
- `TASK_MITRE_MAPPING`
- `RESULTS_THREAT_INTEL`

##### `aiva threat batch <file>`

**選項**:
- `--format json`

**對應 Topics**:
- `TASK_THREAT_INTEL_BATCH`

##### `aiva threat mitre <technique_id>`

**選項**:
- `--format json`

**對應 Topics**:
- `TASK_MITRE_MAPPING`
- `RESULTS_THREAT_INTEL`

