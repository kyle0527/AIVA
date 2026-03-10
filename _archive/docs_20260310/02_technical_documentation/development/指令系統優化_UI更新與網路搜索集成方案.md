# 🔧 指令系統優化 - UI 更新與網路搜索集成方案

**分析日期**: 2025年12月1日  
**目標**: 在保持完整操控的情況下，優化指令系統以支持 UI 面板更新和網路搜索功能

## 📑 目錄

1. [🔍 當前系統架構分析](#當前系統架構分析)
2. [🎯 核心問題識別](#核心問題識別)
3. [💡 優化方案設計](#優化方案設計)
4. [🛤️ 實施路徑](#實施路徑)
5. [🔧 技術細節](#技術細節)
6. [🧪 測試驗證](#測試驗證)

---

---

## 🔍 當前系統架構分析

### 1. 指令系統架構

```
┌─────────────────────────────────────────────────────────────┐
│  User / External System                                      │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  AI Commander V2                                             │
│  - 接收任務請求                                              │
│  - 識別任務領域 (Attack/Defense/Analysis/Training)          │
│  - 初始化 AICommandCenter                                    │
│  - 管理協調器 (Coordinators)                                │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  AICommandCenter (統一命令調度中心)                         │
│  - 接收 AICommand                                            │
│  - 路由到對應模組 CommandHandler                             │
│  - 管理超時和重試                                            │
│  - 返回 AICommandResult                                      │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  CommandHandler 層                                           │
│  - ScanCommandHandler (掃描模組)                            │
│  - FeaturesCommandHandler (功能測試模組)                    │
│  - IntegrationCommandHandler (整合模組)                      │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  執行引擎層                                                  │
│  - Python/TypeScript/Rust/Go 引擎                            │
│  - XSS/SQLi/SSRF/IDOR 檢測器                                │
│  - OSINT/Recon 工具                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2. UI 面板架構

**現有 UI 組件**：
- `services/core/aiva_core/ui_panel/rich_cli.py` - Rich CLI 介面
- `services/core/aiva_core/ui_panel/dashboard.py` - Web 儀表板
- `services/core/aiva_core/ui_panel/improved_ui.py` - 改進版 UI
- `web/js/aiva-dashboard.js` - 前端 JavaScript
- `web/contracts/aiva-web-contracts.ts` - TypeScript 合約

**UI 更新機制缺失**：
- ❌ 沒有實時推送機制
- ❌ 沒有進度回調接口
- ❌ 沒有狀態變更通知
- ❌ 缺少 WebSocket/SSE 連接

### 3. 網路搜索能力現狀

**現有搜索功能**：
- ✅ `SubdomainEnumerator._enumerate_search_engines()` - 搜索引擎枚舉（未實現）
- ✅ `OSINTRecon.search_secrets()` - 敏感信息搜索（本地）
- ✅ `WebRecon` - 網站信息收集（本地爬取）
- ✅ RAG 向量搜索 - 內部知識庫搜索

**缺失的網路搜索**：
- ❌ 外部搜索引擎 API 集成（Google/Bing/DuckDuckGo）
- ❌ ExploitDB 在線搜索
- ❌ CVE/CWE 在線查詢
- ❌ GitHub Issues/StackOverflow 搜索
- ❌ 威脅情報在線查詢

---

## 🎯 核心問題識別

### 問題 1: 指令系統缺少 UI 更新機制

**現象**：
```python
# 當前：AI 下達命令後，UI 無法知道執行進度
command = AICommand(
    command_id="scan_001",
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={"targets": ["http://example.com"]}
)

result = await command_center.execute(command)  # UI 只能等待最終結果
# 問題：UI 在整個過程中無法獲知進度
```

**影響**：
- 用戶體驗差，無法看到實時進度
- 長時間任務缺少反饋，用戶不知道系統是否在工作
- 無法顯示中間結果

### 問題 2: 缺少網路搜索指令類型

**現象**：
```python
# 當前 CommandType 枚舉
class CommandType(str, Enum):
    SCAN_PHASE0 = "scan_phase0"
    SCAN_PHASE1 = "scan_phase1"
    FEATURE_XSS_TEST = "feature_xss_test"
    FEATURE_SQLI_TEST = "feature_sqli_test"
    # ... 其他功能測試
    
    # ❌ 缺少：
    # SEARCH_EXPLOIT_DB = "search_exploit_db"
    # SEARCH_CVE = "search_cve"
    # SEARCH_GOOGLE = "search_google"
    # SEARCH_GITHUB = "search_github"
```

**影響**：
- AI 無法通過統一指令系統進行網路搜索
- 缺少威脅情報和漏洞數據庫的在線查詢能力
- 無法獲取最新的安全資訊

### 問題 3: CommandHandler 缺少實時通信接口

**現象**：
```python
# 當前 CommandHandler 協議
class CommandHandler(Protocol):
    async def handle_command(
        self, 
        command: AICommand, 
        context: Optional[CommandContext] = None
    ) -> AICommandResult:  # 只返回最終結果
        ...

# ❌ 缺少進度回調
# ❌ 缺少狀態更新機制
# ❌ 缺少事件推送接口
```

---

## 💡 優化方案設計

### 方案 1: 增強指令系統支持實時 UI 更新

#### 1.1 擴展 AICommand 支持回調

```python
# 新增：CommandCallback 協議
class CommandCallback(Protocol):
    """指令執行回調接口"""
    
    async def on_progress(
        self, 
        command_id: str,
        progress: float,  # 0.0 - 1.0
        message: str,
        metadata: Dict[str, Any]
    ) -> None:
        """進度更新回調"""
        ...
    
    async def on_status_change(
        self,
        command_id: str,
        old_status: CommandStatus,
        new_status: CommandStatus
    ) -> None:
        """狀態變更回調"""
        ...
    
    async def on_partial_result(
        self,
        command_id: str,
        result_type: str,
        data: Any
    ) -> None:
        """中間結果回調"""
        ...


# 擴展 AICommand 模型
class AICommand(BaseModel):
    """AI 統一指令格式（擴展版）"""
    
    # 原有欄位
    command_id: str
    command_type: CommandType
    target_module: str
    payload: Dict[str, Any]
    
    # ✨ 新增：回調配置
    enable_callbacks: bool = Field(
        default=False,
        description="是否啟用回調機制"
    )
    
    callback_url: Optional[str] = Field(
        default=None,
        description="WebSocket/SSE 回調 URL"
    )
    
    ui_update_interval: float = Field(
        default=1.0,
        description="UI 更新間隔（秒）"
    )
    
    # 回調處理器（僅在 Python 內部使用）
    _callback_handler: Optional[CommandCallback] = PrivateAttr(default=None)
    
    def set_callback(self, callback: CommandCallback):
        """設置回調處理器"""
        self._callback_handler = callback
```

#### 1.2 修改 AICommandCenter 支持回調

```python
class AICommandCenter:
    """AI 命令中心（擴展版）"""
    
    async def execute(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """執行命令（支持回調）"""
        
        start_time = time.time()
        
        try:
            # 通知開始執行
            if command.enable_callbacks and command._callback_handler:
                await command._callback_handler.on_status_change(
                    command.command_id,
                    CommandStatus.PENDING,
                    CommandStatus.RUNNING
                )
            
            # 獲取處理器
            handler = self._handlers.get(command.target_module)
            if not handler:
                raise ValueError(f"未註冊模組: {command.target_module}")
            
            # ✨ 新增：將回調傳遞給處理器
            if command.enable_callbacks and hasattr(handler, 'set_callback'):
                handler.set_callback(command._callback_handler)
            
            # 執行命令
            result = await handler.handle_command(command, context)
            
            # 通知完成
            if command.enable_callbacks and command._callback_handler:
                await command._callback_handler.on_status_change(
                    command.command_id,
                    CommandStatus.RUNNING,
                    result.status
                )
            
            return result
            
        except Exception as e:
            logger.error(f"命令執行失敗: {e}")
            
            if command.enable_callbacks and command._callback_handler:
                await command._callback_handler.on_status_change(
                    command.command_id,
                    CommandStatus.RUNNING,
                    CommandStatus.FAILED
                )
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
```

#### 1.3 實現 UI 回調處理器

```python
# 新文件：services/core/aiva_core/ui_panel/command_callback.py
class UICommandCallback:
    """UI 命令回調處理器"""
    
    def __init__(self, websocket_manager: Optional['WebSocketManager'] = None):
        self.websocket_manager = websocket_manager
        self.progress_cache: Dict[str, float] = {}
        self.logger = get_logger(__name__)
    
    async def on_progress(
        self,
        command_id: str,
        progress: float,
        message: str,
        metadata: Dict[str, Any]
    ) -> None:
        """進度更新 - 推送到 UI"""
        
        self.progress_cache[command_id] = progress
        
        update_message = {
            "type": "progress_update",
            "command_id": command_id,
            "progress": progress,
            "message": message,
            "metadata": metadata,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        # 推送到 WebSocket 客戶端
        if self.websocket_manager:
            await self.websocket_manager.broadcast(update_message)
        
        # 更新 Rich CLI
        if hasattr(console, 'update_progress'):
            console.update_progress(command_id, progress, message)
        
        self.logger.info(f"[UI Update] {command_id}: {progress:.1%} - {message}")
    
    async def on_status_change(
        self,
        command_id: str,
        old_status: CommandStatus,
        new_status: CommandStatus
    ) -> None:
        """狀態變更 - 更新 UI"""
        
        status_message = {
            "type": "status_change",
            "command_id": command_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast(status_message)
        
        self.logger.info(
            f"[Status Change] {command_id}: {old_status.value} → {new_status.value}"
        )
    
    async def on_partial_result(
        self,
        command_id: str,
        result_type: str,
        data: Any
    ) -> None:
        """中間結果 - 顯示在 UI"""
        
        result_message = {
            "type": "partial_result",
            "command_id": command_id,
            "result_type": result_type,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        if self.websocket_manager:
            await self.websocket_manager.broadcast(result_message)
        
        # 在 Rich CLI 顯示中間結果
        if result_type == "vulnerability_found":
            console.print(
                f"[aiva.warning]🚨 發現漏洞[/aiva.warning]: {data.get('type', 'Unknown')}"
            )
        elif result_type == "url_discovered":
            console.print(
                f"[aiva.info]🔗 發現 URL[/aiva.info]: {data.get('url', 'N/A')}"
            )
```

### 方案 2: 增加網路搜索指令類型

#### 2.1 擴展 CommandType 枚舉

```python
# 修改：services/aiva_common/schemas/commands.py
class CommandType(str, Enum):
    """AI 命令類型（擴展版）"""
    
    # ===== 現有命令 =====
    # Scan 模組命令
    SCAN_PHASE0 = "scan_phase0"
    SCAN_PHASE1 = "scan_phase1"
    SCAN_COMPREHENSIVE = "scan_comprehensive"
    
    # Features 模組命令
    FEATURE_XSS_TEST = "feature_xss_test"
    FEATURE_SQLI_TEST = "feature_sqli_test"
    FEATURE_SSRF_TEST = "feature_ssrf_test"
    FEATURE_IDOR_TEST = "feature_idor_test"
    
    # ===== ✨ 新增：網路搜索命令 =====
    # 漏洞數據庫搜索
    SEARCH_EXPLOIT_DB = "search_exploit_db"        # ExploitDB 搜索
    SEARCH_CVE_DETAILS = "search_cve_details"      # CVE Details 搜索
    SEARCH_CWE_INFO = "search_cwe_info"            # CWE 信息查詢
    SEARCH_CAPEC_PATTERNS = "search_capec_patterns"  # CAPEC 攻擊模式
    
    # 威脅情報搜索
    SEARCH_THREAT_INTEL = "search_threat_intel"    # 威脅情報查詢
    SEARCH_IOC = "search_ioc"                      # IOC（入侵指標）搜索
    SEARCH_MALWARE_ANALYSIS = "search_malware_analysis"  # 惡意軟件分析
    
    # 開源情報搜索 (OSINT)
    SEARCH_GOOGLE = "search_google"                # Google 搜索
    SEARCH_DUCKDUCKGO = "search_duckduckgo"        # DuckDuckGo 搜索
    SEARCH_GITHUB = "search_github"                # GitHub 代碼/Issue 搜索
    SEARCH_STACKOVERFLOW = "search_stackoverflow"   # StackOverflow 搜索
    SEARCH_SHODAN = "search_shodan"                # Shodan 設備搜索
    SEARCH_CENSYS = "search_censys"                # Censys 資產搜索
    
    # 社交工程搜索
    SEARCH_EMAIL_BREACH = "search_email_breach"    # 郵箱洩露查詢
    SEARCH_DOMAIN_INFO = "search_domain_info"      # 域名信息查詢
    SEARCH_WHOIS = "search_whois"                  # WHOIS 查詢
    
    # AI 輔助搜索
    RAG_SEARCH_INTERNAL = "rag_search_internal"    # RAG 內部知識庫搜索
    RAG_SEARCH_EXTERNAL = "rag_search_external"    # RAG 外部網路搜索
```

#### 2.2 創建 SearchCommandHandler

```python
# 新文件：services/integration/search_command_handler.py
from typing import Dict, Any, List, Optional
import aiohttp
import asyncio
from datetime import datetime, UTC

from services.aiva_common.schemas import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext
)
from services.aiva_common.command_center import CommandHandler
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class SearchCommandHandler(CommandHandler):
    """網路搜索命令處理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # API 配置
        self.google_api_key = self.config.get("google_api_key")
        self.shodan_api_key = self.config.get("shodan_api_key")
        self.github_token = self.config.get("github_token")
        
        # 搜索引擎實例
        self.search_engines = {
            "google": self._search_google,
            "duckduckgo": self._search_duckduckgo,
            "github": self._search_github,
            "exploitdb": self._search_exploitdb,
            "cve": self._search_cve_details,
            "shodan": self._search_shodan,
        }
        
        self.callback_handler = None
        
        logger.info("✅ SearchCommandHandler 已初始化")
    
    def set_callback(self, callback):
        """設置回調處理器"""
        self.callback_handler = callback
    
    async def handle_command(
        self,
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理搜索命令"""
        
        start_time = time.time()
        
        try:
            # 路由到對應的搜索方法
            if command.command_type == CommandType.SEARCH_GOOGLE:
                results = await self._search_google(command.payload)
            
            elif command.command_type == CommandType.SEARCH_DUCKDUCKGO:
                results = await self._search_duckduckgo(command.payload)
            
            elif command.command_type == CommandType.SEARCH_GITHUB:
                results = await self._search_github(command.payload)
            
            elif command.command_type == CommandType.SEARCH_EXPLOIT_DB:
                results = await self._search_exploitdb(command.payload)
            
            elif command.command_type == CommandType.SEARCH_CVE_DETAILS:
                results = await self._search_cve_details(command.payload)
            
            elif command.command_type == CommandType.SEARCH_SHODAN:
                results = await self._search_shodan(command.payload)
            
            elif command.command_type == CommandType.SEARCH_THREAT_INTEL:
                results = await self._search_threat_intel(command.payload)
            
            else:
                return AICommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.FAILED,
                    error=f"不支持的搜索類型: {command.command_type}",
                    execution_time=time.time() - start_time
                )
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                result=results,
                execution_time=time.time() - start_time,
                metadata={
                    "search_type": command.command_type.value,
                    "results_count": len(results.get("items", []))
                }
            )
            
        except Exception as e:
            logger.error(f"搜索命令執行失敗: {e}")
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    # ===== 搜索引擎實現 =====
    
    async def _search_google(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Google 搜索"""
        query = payload.get("query", "")
        max_results = payload.get("max_results", 10)
        
        if not self.google_api_key:
            raise ValueError("Google API Key 未配置")
        
        # 使用 Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.config.get("google_search_engine_id"),
            "q": query,
            "num": min(max_results, 10)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                return {
                    "query": query,
                    "total_results": data.get("searchInformation", {}).get("totalResults", 0),
                    "items": [
                        {
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "snippet": item.get("snippet")
                        }
                        for item in data.get("items", [])
                    ],
                    "search_time": data.get("searchInformation", {}).get("searchTime", 0)
                }
    
    async def _search_duckduckgo(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """DuckDuckGo 搜索（使用 HTML 爬取）"""
        query = payload.get("query", "")
        max_results = payload.get("max_results", 10)
        
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                return {
                    "query": query,
                    "abstract": data.get("Abstract", ""),
                    "abstract_url": data.get("AbstractURL", ""),
                    "related_topics": [
                        {
                            "text": topic.get("Text", ""),
                            "url": topic.get("FirstURL", "")
                        }
                        for topic in data.get("RelatedTopics", [])[:max_results]
                    ]
                }
    
    async def _search_github(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """GitHub 搜索"""
        query = payload.get("query", "")
        search_type = payload.get("search_type", "repositories")  # repositories, code, issues
        max_results = payload.get("max_results", 10)
        
        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        
        url = f"https://api.github.com/search/{search_type}"
        params = {
            "q": query,
            "per_page": min(max_results, 100)
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                return {
                    "query": query,
                    "search_type": search_type,
                    "total_count": data.get("total_count", 0),
                    "items": [
                        {
                            "name": item.get("name", item.get("title", "")),
                            "url": item.get("html_url", ""),
                            "description": item.get("description", ""),
                            "score": item.get("score", 0)
                        }
                        for item in data.get("items", [])
                    ]
                }
    
    async def _search_exploitdb(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """ExploitDB 搜索"""
        query = payload.get("query", "")
        platform = payload.get("platform", "")  # windows, linux, web
        
        # ExploitDB 的 API 端點
        url = "https://www.exploit-db.com/search"
        params = {
            "q": query,
            "platform": platform
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                # 解析 HTML 或使用 API（需要實際實現）
                # 這裡提供簡化版本
                
                return {
                    "query": query,
                    "platform": platform,
                    "exploits": [
                        # 實際需要爬取或使用 API
                    ]
                }
    
    async def _search_cve_details(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """CVE Details 搜索"""
        cve_id = payload.get("cve_id", "")
        keyword = payload.get("keyword", "")
        
        if cve_id:
            # 查詢特定 CVE
            url = f"https://cvedetails.com/cve/{cve_id}/"
        else:
            # 關鍵字搜索
            url = "https://cvedetails.com/search.php"
        
        # 實際需要使用 NVD API: https://nvd.nist.gov/developers
        return {
            "query": cve_id or keyword,
            "cve_details": []
        }
    
    async def _search_shodan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Shodan 設備搜索"""
        query = payload.get("query", "")
        
        if not self.shodan_api_key:
            raise ValueError("Shodan API Key 未配置")
        
        url = "https://api.shodan.io/shodan/host/search"
        params = {
            "key": self.shodan_api_key,
            "query": query
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                return {
                    "query": query,
                    "total": data.get("total", 0),
                    "matches": [
                        {
                            "ip": match.get("ip_str"),
                            "port": match.get("port"),
                            "org": match.get("org"),
                            "data": match.get("data", "")[:200]  # 前200字符
                        }
                        for match in data.get("matches", [])
                    ]
                }
    
    async def _search_threat_intel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """威脅情報搜索（整合多個來源）"""
        indicator = payload.get("indicator", "")  # IP, domain, hash
        indicator_type = payload.get("type", "ip")
        
        # 並行查詢多個威脅情報源
        tasks = [
            self._query_virustotal(indicator, indicator_type),
            self._query_abuseipdb(indicator, indicator_type),
            self._query_alienvault_otx(indicator, indicator_type)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "indicator": indicator,
            "type": indicator_type,
            "sources": {
                "virustotal": results[0] if not isinstance(results[0], Exception) else None,
                "abuseipdb": results[1] if not isinstance(results[1], Exception) else None,
                "alienvault": results[2] if not isinstance(results[2], Exception) else None
            }
        }
    
    async def _query_virustotal(self, indicator: str, itype: str) -> Dict:
        """VirusTotal 查詢"""
        # 需要 VirusTotal API Key
        return {}
    
    async def _query_abuseipdb(self, indicator: str, itype: str) -> Dict:
        """AbuseIPDB 查詢"""
        # 需要 AbuseIPDB API Key
        return {}
    
    async def _query_alienvault_otx(self, indicator: str, itype: str) -> Dict:
        """AlienVault OTX 查詢"""
        # 使用 AlienVault OTX API
        return {}
```

#### 2.3 註冊 SearchCommandHandler

```python
# 修改：services/core/aiva_core/task_planning/ai_commander_v2.py
class AICommanderV2:
    """AI 指揮官 V2（擴展版）"""
    
    async def initialize(self) -> bool:
        """初始化 AI Commander"""
        try:
            logger.info("Initializing AI Commander V2...")
            
            # ... 現有初始化代碼 ...
            
            # ✨ 新增：註冊搜索模組處理器
            from services.integration.search_command_handler import SearchCommandHandler
            
            search_config = {
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
                "google_search_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                "shodan_api_key": os.getenv("SHODAN_API_KEY"),
                "github_token": os.getenv("GITHUB_TOKEN"),
            }
            
            search_handler = SearchCommandHandler(config=search_config)
            self.command_center.register_module("search", search_handler)
            logger.info("✅ 已註冊 search 模組處理器")
            
            self.initialized = True
            logger.info("AI Commander V2 initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"初始化失敗: {e}")
            return False
```

### 方案 3: UI 面板集成

#### 3.1 WebSocket Manager

```python
# 新文件：services/core/aiva_core/ui_panel/websocket_manager.py
from typing import Set, Dict, Any
import asyncio
import json
from datetime import datetime, UTC

from fastapi import WebSocket
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """WebSocket 連接管理器"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logger
    
    async def connect(self, websocket: WebSocket):
        """接受新的 WebSocket 連接"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket 連接建立，當前連接數: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """斷開 WebSocket 連接"""
        self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket 連接斷開，當前連接數: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """發送個人消息"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"發送個人消息失敗: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """廣播消息到所有連接"""
        if not self.active_connections:
            return
        
        # 添加時間戳
        message["timestamp"] = datetime.now(UTC).isoformat()
        
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"廣播消息失敗: {e}")
                disconnected.add(connection)
        
        # 清理斷開的連接
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_progress(
        self,
        command_id: str,
        progress: float,
        message: str,
        metadata: Dict[str, Any] = None
    ):
        """廣播進度更新"""
        await self.broadcast({
            "type": "progress_update",
            "command_id": command_id,
            "progress": progress,
            "message": message,
            "metadata": metadata or {}
        })
    
    async def broadcast_status_change(
        self,
        command_id: str,
        new_status: str,
        metadata: Dict[str, Any] = None
    ):
        """廣播狀態變更"""
        await self.broadcast({
            "type": "status_change",
            "command_id": command_id,
            "status": new_status,
            "metadata": metadata or {}
        })
    
    async def broadcast_result(
        self,
        command_id: str,
        result_type: str,
        data: Any
    ):
        """廣播執行結果"""
        await self.broadcast({
            "type": "result",
            "command_id": command_id,
            "result_type": result_type,
            "data": data
        })


# 全局 WebSocket Manager 實例
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """獲取全局 WebSocket Manager"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager
```

#### 3.2 FastAPI WebSocket 端點

```python
# 新文件：api/websocket_routes.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.core.aiva_core.ui_panel.websocket_manager import get_websocket_manager

router = APIRouter()
websocket_manager = get_websocket_manager()


@router.websocket("/ws/commands")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端點 - 實時命令更新"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # 接收客戶端消息（心跳檢查）
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
```

#### 3.3 前端 WebSocket 客戶端

```javascript
// 新文件：web/js/aiva-websocket.js
class AIVAWebSocketClient {
    constructor(url = 'ws://localhost:8000/ws/commands') {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 5000;
        this.handlers = {
            'progress_update': [],
            'status_change': [],
            'result': []
        };
        
        this.connect();
    }
    
    connect() {
        console.log('連接 WebSocket:', this.url);
        
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('✅ WebSocket 已連接');
            
            // 發送心跳
            this.startHeartbeat();
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ WebSocket 錯誤:', error);
        };
        
        this.ws.onclose = () => {
            console.log('🔌 WebSocket 已斷開');
            
            // 自動重連
            setTimeout(() => this.connect(), this.reconnectInterval);
        };
    }
    
    startHeartbeat() {
        setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 30000);  // 每 30 秒發送一次心跳
    }
    
    handleMessage(data) {
        console.log('📨 收到消息:', data);
        
        const type = data.type;
        const handlers = this.handlers[type] || [];
        
        handlers.forEach(handler => {
            try {
                handler(data);
            } catch (error) {
                console.error('處理消息失敗:', error);
            }
        });
    }
    
    on(event, handler) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(handler);
    }
    
    updateProgress(commandId, progress, message) {
        // 更新 UI 進度條
        const progressBar = document.getElementById(`progress-${commandId}`);
        if (progressBar) {
            progressBar.style.width = `${progress * 100}%`;
            progressBar.innerText = `${(progress * 100).toFixed(1)}%`;
        }
        
        const statusText = document.getElementById(`status-${commandId}`);
        if (statusText) {
            statusText.innerText = message;
        }
    }
    
    updateStatus(commandId, status) {
        const statusBadge = document.getElementById(`badge-${commandId}`);
        if (statusBadge) {
            statusBadge.className = `status-badge status-${status}`;
            statusBadge.innerText = status;
        }
    }
    
    displayResult(commandId, resultType, data) {
        const resultsContainer = document.getElementById('results-container');
        
        if (!resultsContainer) return;
        
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.innerHTML = `
            <h3>${resultType}</h3>
            <pre>${JSON.stringify(data, null, 2)}</pre>
            <small>Command ID: ${commandId}</small>
        `;
        
        resultsContainer.insertBefore(resultCard, resultsContainer.firstChild);
    }
}

// 初始化全局 WebSocket 客戶端
const aivaWS = new AIVAWebSocketClient();

// 註冊事件處理器
aivaWS.on('progress_update', (data) => {
    aivaWS.updateProgress(data.command_id, data.progress, data.message);
});

aivaWS.on('status_change', (data) => {
    aivaWS.updateStatus(data.command_id, data.status);
});

aivaWS.on('result', (data) => {
    aivaWS.displayResult(data.command_id, data.result_type, data.data);
});
```

---

## 🚀 實施路徑

### 階段 1: 基礎設施準備（2-3 天）

**任務清單**：
1. ✅ 創建 `CommandCallback` 協議
2. ✅ 擴展 `AICommand` 模型支持回調
3. ✅ 修改 `AICommandCenter.execute()` 支持回調
4. ✅ 創建 `UICommandCallback` 實現
5. ✅ 實現 `WebSocketManager`
6. ✅ 添加 FastAPI WebSocket 端點
7. ✅ 創建前端 WebSocket 客戶端

**驗證方式**：
```python
# 測試回調機制
async def test_callback():
    # 初始化
    command_center = AICommandCenter()
    callback = UICommandCallback()
    
    # 創建命令
    command = AICommand(
        command_id="test_001",
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={"targets": ["http://example.com"]},
        enable_callbacks=True
    )
    command.set_callback(callback)
    
    # 執行（應該能看到實時更新）
    result = await command_center.execute(command)
```

### 階段 2: 網路搜索集成（3-4 天）

**任務清單**：
1. ✅ 擴展 `CommandType` 添加搜索命令
2. ✅ 創建 `SearchCommandHandler`
3. ✅ 實現各個搜索引擎接口
   - Google Custom Search API
   - DuckDuckGo API
   - GitHub API
   - ExploitDB 爬蟲
   - CVE Details / NVD API
   - Shodan API
4. ✅ 在 `AICommanderV2` 註冊搜索處理器
5. ✅ 配置 API Keys（環境變量）

**驗證方式**：
```python
# 測試搜索功能
async def test_search():
    command_center = AICommandCenter()
    
    # Google 搜索
    command = AICommand(
        command_id="search_google_001",
        command_type=CommandType.SEARCH_GOOGLE,
        target_module="search",
        payload={
            "query": "SQL injection vulnerability example",
            "max_results": 10
        }
    )
    
    result = await command_center.execute(command)
    print(f"找到 {len(result.result['items'])} 個結果")
    
    # GitHub 搜索
    command = AICommand(
        command_id="search_github_001",
        command_type=CommandType.SEARCH_GITHUB,
        target_module="search",
        payload={
            "query": "XSS vulnerability",
            "search_type": "code",
            "max_results": 20
        }
    )
    
    result = await command_center.execute(command)
    print(f"GitHub 找到 {result.result['total_count']} 個結果")
```

### 階段 3: UI 面板整合（2-3 天）

**任務清單**：
1. ✅ 修改 Rich CLI 支持實時更新
2. ✅ 更新 Dashboard 集成 WebSocket
3. ✅ 前端頁面添加實時進度顯示
4. ✅ 添加搜索結果展示組件
5. ✅ 測試端到端流程

**前端實時進度組件**：
```html
<!-- 新增：實時進度面板 -->
<div class="command-progress-panel">
    <h3>執行中的命令</h3>
    
    <div id="active-commands">
        <!-- 動態生成 -->
        <div class="command-card" data-command-id="scan_001">
            <div class="command-header">
                <span class="command-type">SCAN_PHASE0</span>
                <span class="status-badge status-running" id="badge-scan_001">Running</span>
            </div>
            
            <div class="command-body">
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-scan_001" style="width: 45%;">45%</div>
                </div>
                <p class="status-text" id="status-scan_001">正在掃描 URL: http://example.com/api</p>
            </div>
            
            <div class="command-footer">
                <small>開始時間: 2025-12-01 10:30:15</small>
            </div>
        </div>
    </div>
</div>

<div class="search-results-panel">
    <h3>搜索結果</h3>
    
    <div id="results-container">
        <!-- WebSocket 推送的結果會顯示在這裡 -->
    </div>
</div>
```

### 階段 4: 完整測試與優化（2-3 天）

**測試項目**：
1. 性能測試
   - 單個命令執行性能
   - 並發命令執行測試
   - WebSocket 連接穩定性
   - 搜索 API 響應時間

2. 功能測試
   - 所有搜索命令類型
   - 回調機制準確性
   - UI 實時更新正確性
   - 錯誤處理和重試

3. 整合測試
   - AI → 搜索 → UI 完整流程
   - 多用戶同時使用
   - 長時間運行穩定性

---

## 📊 預期效果

### 1. UI 實時更新

**Before (現狀)**：
```
用戶: 開始掃描
UI: 正在執行...（轉圈圈）
[等待 10 分鐘]
UI: 完成！（顯示結果）
```

**After (優化後)**：
```
用戶: 開始掃描
UI: 初始化引擎... [0%]
UI: 正在爬取首頁... [10%]
UI: 發現 25 個 URL... [25%]
UI: 正在測試 XSS... [40%]
UI: ⚠️ 發現 XSS 漏洞！ [45%]
UI: 正在測試 SQL 注入... [60%]
UI: 完成掃描！ [100%]
```

### 2. 網路搜索能力

**AI 決策流程**：
```
AI: 發現未知漏洞類型，需要搜索類似案例

# 執行 Google 搜索
search_result = await command_center.execute(AICommand(
    command_type=CommandType.SEARCH_GOOGLE,
    payload={"query": "CVE-2024-XXXX exploitation"}
))

# 執行 GitHub 搜索
github_result = await command_center.execute(AICommand(
    command_type=CommandType.SEARCH_GITHUB,
    payload={"query": "CVE-2024-XXXX POC", "search_type": "code"}
))

# 執行 ExploitDB 搜索
exploitdb_result = await command_center.execute(AICommand(
    command_type=CommandType.SEARCH_EXPLOIT_DB,
    payload={"query": "CVE-2024-XXXX"}
))

AI: 綜合搜索結果，制定攻擊計劃...
```

### 3. 完整操控能力

**保持完整控制**：
```python
# AI Commander 完全掌控所有流程
class AICommanderV2:
    async def execute_attack_task(self, task):
        # 1. 決策：需要搜索背景資訊
        if self._need_background_research(task):
            search_results = await self._research_target(task)
        
        # 2. 規劃：根據搜索結果調整計劃
        plan = await self._create_attack_plan(task, search_results)
        
        # 3. 執行：下達命令並監控進度
        for step in plan.steps:
            command = self._step_to_command(step)
            command.enable_callbacks = True  # 啟用實時更新
            
            # 執行並實時更新 UI
            result = await self.command_center.execute(command)
            
            # 4. 分析：評估結果並決定下一步
            next_action = await self._analyze_result(result)
            
            if next_action == "search_more":
                # 需要更多資訊，繼續搜索
                additional_info = await self._search_for_more_info(result)
        
        # 5. 報告：整合所有結果
        return self._generate_final_report()
```

---

## ✅ 優勢總結

### 1. **保持完整操控**
- AI Commander 仍然是唯一的指揮中心
- 所有命令都通過 AICommandCenter 統一調度
- 完整的執行歷史和追蹤能力

### 2. **實時 UI 更新**
- WebSocket 推送實時進度
- 用戶隨時了解執行狀態
- 可以中途取消或調整任務

### 3. **強大的網路搜索**
- 整合多個搜索引擎和情報源
- AI 可以自主搜索補充知識
- 支持威脅情報和漏洞數據庫查詢

### 4. **易於擴展**
- 新增搜索源只需實現一個方法
- 回調機制可以適配任何 UI
- 命令類型可以無限擴展

### 5. **類型安全**
- 所有命令都是 Pydantic 模型
- 編譯時類型檢查
- API 文檔自動生成

---

## 🔧 配置文件範例

```yaml
# config/aiva_search.yaml
search:
  google:
    api_key: "${GOOGLE_API_KEY}"
    search_engine_id: "${GOOGLE_SEARCH_ENGINE_ID}"
    max_results: 10
    rate_limit: 100  # 每天最多 100 次查詢
  
  github:
    token: "${GITHUB_TOKEN}"
    max_results: 50
    rate_limit: 5000  # 每小時最多 5000 次請求
  
  shodan:
    api_key: "${SHODAN_API_KEY}"
    max_results: 100
  
  exploitdb:
    base_url: "https://www.exploit-db.com"
    cache_ttl: 3600  # 緩存 1 小時
  
  cve_details:
    api_url: "https://services.nvd.nist.gov/rest/json/cves/2.0"
    api_key: "${NVD_API_KEY}"  # 可選，但有 key 可以提高速率限制

ui:
  websocket:
    url: "ws://localhost:8000/ws/commands"
    reconnect_interval: 5000  # 毫秒
    heartbeat_interval: 30000  # 毫秒
  
  progress_update:
    interval: 1.0  # 秒
    show_intermediate_results: true
    max_history: 100  # 最多保存 100 條歷史

command_center:
  default_timeout: 300  # 秒
  max_retries: 3
  retry_delay: 5  # 秒
  enable_history: true
  max_history: 1000
```

---

## 📝 環境變量設置

```bash
# .env 文件
# Google Search API
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# GitHub API
GITHUB_TOKEN=ghp_your_github_token_here

# Shodan API
SHODAN_API_KEY=your_shodan_api_key_here

# NVD (National Vulnerability Database) API
NVD_API_KEY=your_nvd_api_key_here

# VirusTotal API (可選)
VIRUSTOTAL_API_KEY=your_virustotal_api_key_here

# AbuseIPDB API (可選)
ABUSEIPDB_API_KEY=your_abuseipdb_api_key_here
```

---

## 🎯 總結

此優化方案通過以下方式實現目標：

1. **擴展指令系統** - 添加回調機制和搜索命令類型
2. **集成網路搜索** - 創建 SearchCommandHandler 整合多個搜索源
3. **實時 UI 更新** - 通過 WebSocket 推送執行進度和結果
4. **保持完整操控** - AI Commander 仍然是唯一的指揮中心

**所有修改都是向後兼容的，不會影響現有功能的運行。**

實施後，AI 將能夠：
- 📊 實時更新 UI，讓用戶了解執行進度
- 🔍 搜索外部資源補充知識
- 🎯 根據搜索結果動態調整策略
- 🚀 保持對整個系統的完整控制
