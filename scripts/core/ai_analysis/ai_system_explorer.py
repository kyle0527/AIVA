#!/usr/bin/env python3
"""
AIVA AI 系統自我探索器
========================

設計原則:
- AI 可插拔式: 移除此模組後，系統仍可完全手動操作
- 非侵入性: 不修改現有模組，只進行讀取和分析
- CLI 友好: 為 CLI 指令生成提供完整的系統狀態信息
- 模組化: 每個模組可獨立檢測和診斷

核心功能:
1. 五大模組自動掃描和能力檢測
2. 模組健康度和性能指標分析
3. 模組間通信狀態診斷
4. 問題自動識別和分類（僅識別，不自動修復）
5. 優化建議自動生成
6. 為 CLI 指令生成提供基礎數據

作者: AIVA AI Assistant
日期: 2025-10-28
版本: 1.0.0
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import subprocess
import psutil
import platform
import importlib.util

# 加入 AIVA 路徑
sys.path.insert(0, str(Path(__file__).parent))

try:
    from services.aiva_common.utils.logging import get_logger
    from services.aiva_common.utils.ids import new_id
    from services.aiva_common.enums import ProgrammingLanguage, Severity
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def new_id(prefix=""):
        import uuid
        return f"{prefix}_{str(uuid.uuid4())[:8]}"


@dataclass
class ModuleInfo:
    """模組基本信息"""
    name: str
    path: str
    language: str
    file_count: int
    line_count: int
    size_bytes: int
    last_modified: datetime
    dependencies: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)


@dataclass
class HealthStatus:
    """健康狀態"""
    status: str  # healthy, warning, critical, unknown
    score: float  # 0.0 - 1.0
    checks: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """性能指標"""
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    response_time: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    availability: Optional[float] = None


@dataclass
class ModuleDiagnostic:
    """模組診斷結果"""
    module_info: ModuleInfo
    health_status: HealthStatus
    performance_metrics: PerformanceMetrics
    communication_status: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    cli_commands: List[str] = field(default_factory=list)
    manual_checks: List[str] = field(default_factory=list)


@dataclass 
class SystemDiagnosticReport:
    """系統診斷報告"""
    report_id: str
    timestamp: datetime
    system_info: Dict[str, Any]
    modules: Dict[str, ModuleDiagnostic]
    overall_health: HealthStatus
    critical_issues: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)


class SystemSelfExplorer:
    """AIVA 系統自我探索器 - 可插拔式 AI 組件"""
    
    # 五大模組定義
    CORE_MODULES = {
        "ai_core": {
            "name": "AI 核心引擎",
            "path": "services/core/aiva_core",
            "entry_points": ["dialog/assistant.py", "training/training_orchestrator.py"],
            "languages": ["Python"],
            "dependencies": ["torch", "transformers", "openai"]
        },
        "attack_engine": {
            "name": "攻擊執行引擎", 
            "path": "services/core/aiva_core/attack",
            "entry_points": ["exploit_manager.py", "attack_chain.py"],
            "languages": ["Python"],
            "dependencies": ["requests", "sqlparse"]
        },
        "scan_engine": {
            "name": "掃描引擎",
            "path": "services/scan",
            "entry_points": ["aiva_scan", "info_gatherer_rust"],
            "languages": ["Python", "Rust"],
            "dependencies": ["rust", "cargo", "regex"]
        },
        "integration_service": {
            "name": "整合服務",
            "path": "services/integration", 
            "entry_points": ["capability", "notification"],
            "languages": ["Python"],
            "dependencies": ["pika", "redis", "celery"]
        },
        "feature_detection": {
            "name": "功能檢測",
            "path": "services/features",
            "entry_points": ["function_*"],
            "languages": ["Python", "Go", "Rust"],
            "dependencies": ["go", "cargo", "requests"]
        }
    }
    
    def __init__(self, workspace_root: Optional[Path] = None):
        """初始化系統探索器"""
        self.workspace_root = workspace_root or Path.cwd()
        self.exploration_id = new_id("explore")
        self.start_time = datetime.now()
        
        # 確保可插拔性 - 即使沒有依賴也能運行
        self.capabilities = self._check_optional_capabilities()
        
        logger.info(f"🔍 AIVA 系統自我探索器已初始化 (ID: {self.exploration_id})")
        logger.info(f"📁 工作目錄: {self.workspace_root}")
        logger.info(f"🔧 可用能力: {list(self.capabilities.keys())}")
    
    def _check_optional_capabilities(self) -> Dict[str, bool]:
        """檢查可選功能的可用性"""
        capabilities = {}
        
        # 檢查系統監控能力
        try:
            import psutil
            capabilities["system_monitoring"] = True
        except ImportError:
            capabilities["system_monitoring"] = False
            logger.warning("psutil 未安裝，系統監控功能受限")
        
        # 檢查 Git 能力  
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            capabilities["git"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            capabilities["git"] = False
        
        # 檢查多語言支援
        for lang in ["python", "go", "cargo"]:
            try:
                cmd = {"python": "python --version", "go": "go version", "cargo": "cargo --version"}
                subprocess.run(cmd[lang].split(), capture_output=True, check=True)
                capabilities[lang] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                capabilities[lang] = False
        
        return capabilities
    
    async def explore_system(self, detailed: bool = True) -> SystemDiagnosticReport:
        """執行完整系統探索 - 主要入口點"""
        logger.info("🚀 開始系統自我探索...")
        
        try:
            # 收集系統基本信息
            system_info = self._collect_system_info()
            
            # 探索五大模組
            module_diagnostics = {}
            for module_id, module_config in self.CORE_MODULES.items():
                logger.info(f"🔍 探索模組: {module_config['name']}")
                
                try:
                    diagnostic = await self._explore_module(module_id, module_config, detailed)
                    module_diagnostics[module_id] = diagnostic
                except Exception as e:
                    logger.error(f"❌ 模組 {module_id} 探索失敗: {str(e)}")
                    # 創建錯誤診斷
                    module_diagnostics[module_id] = self._create_error_diagnostic(
                        module_id, module_config, str(e)
                    )
            
            # 生成整體健康狀態
            overall_health = self._calculate_overall_health(module_diagnostics)
            
            # 生成系統級建議
            critical_issues, optimization_suggestions, next_actions = \
                self._generate_system_recommendations(module_diagnostics, overall_health)
            
            # 創建診斷報告
            report = SystemDiagnosticReport(
                report_id=self.exploration_id,
                timestamp=self.start_time,
                system_info=system_info,
                modules=module_diagnostics,
                overall_health=overall_health,
                critical_issues=critical_issues,
                optimization_suggestions=optimization_suggestions,
                next_actions=next_actions
            )
            
            # 保存報告
            await self._save_diagnostic_report(report)
            
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"✅ 系統探索完成 (耗時: {duration:.2f}s)")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ 系統探索失敗: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系統基本信息"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture(),
            "workspace_root": str(self.workspace_root),
            "exploration_time": self.start_time.isoformat(),
            "capabilities": self.capabilities
        }
        
        # 如果有 psutil，收集系統資源信息
        if self.capabilities.get("system_monitoring"):
            try:
                info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "disk_total": psutil.disk_usage(str(self.workspace_root)).total,
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
                })
            except Exception as e:
                logger.warning(f"收集系統資源信息失敗: {e}")
        
        return info
    
    async def _explore_module(
        self, 
        module_id: str, 
        module_config: Dict[str, Any], 
        detailed: bool = True
    ) -> ModuleDiagnostic:
        """探索單個模組"""
        
        # 1. 收集模組基本信息
        module_info = await self._collect_module_info(module_id, module_config)
        
        # 2. 執行健康檢查
        health_status = await self._check_module_health(module_info, module_config)
        
        # 3. 收集性能指標（如果可用）
        performance_metrics = await self._collect_performance_metrics(module_info)
        
        # 4. 檢查通信狀態
        communication_status = await self._check_communication_status(module_info, module_config)
        
        # 5. 生成建議和 CLI 指令
        recommendations, cli_commands, manual_checks = self._generate_module_recommendations(
            module_info, health_status, performance_metrics, communication_status
        )
        
        return ModuleDiagnostic(
            module_info=module_info,
            health_status=health_status,
            performance_metrics=performance_metrics,
            communication_status=communication_status,
            recommendations=recommendations,
            cli_commands=cli_commands,
            manual_checks=manual_checks
        )
    
    async def _collect_module_info(self, module_id: str, module_config: Dict[str, Any]) -> ModuleInfo:
        """收集模組基本信息"""
        module_path = self.workspace_root / module_config["path"]
        
        if not module_path.exists():
            return ModuleInfo(
                name=module_config["name"],
                path=str(module_path),
                language="Unknown",
                file_count=0,
                line_count=0,
                size_bytes=0,
                last_modified=datetime.min,
                dependencies=[],
                entry_points=[],
                config_files=[]
            )
        
        # 統計文件信息
        file_count = 0
        line_count = 0
        size_bytes = 0
        last_modified = datetime.min
        config_files = []
        
        # 掃描目錄
        for file_path in module_path.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    file_count += 1
                    size_bytes += stat.st_size
                    file_modified = datetime.fromtimestamp(stat.st_mtime)
                    if file_modified > last_modified:
                        last_modified = file_modified
                    
                    # 計算行數（僅文本文件）
                    if file_path.suffix in ['.py', '.go', '.rs', '.yaml', '.yml', '.json', '.toml']:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                line_count += len(f.readlines())
                        except (UnicodeDecodeError, PermissionError):
                            pass
                    
                    # 識別配置文件
                    if file_path.name in ['config.yaml', 'config.yml', 'settings.py', 'Cargo.toml', 'go.mod', 'package.json']:
                        config_files.append(str(file_path.relative_to(self.workspace_root)))
                        
                except (PermissionError, OSError):
                    continue
        
        # 檢測主要語言
        languages = module_config.get("languages", ["Python"])
        primary_language = languages[0] if languages else "Unknown"
        
        # 檢測依賴關係
        dependencies = await self._detect_dependencies(module_path, primary_language)
        
        # 檢測入口點
        entry_points = []
        for entry_pattern in module_config.get("entry_points", []):
            if "*" in entry_pattern:
                # 通配符模式
                pattern_parts = entry_pattern.split("*")
                for file_path in module_path.rglob("*"):
                    if all(part in str(file_path) for part in pattern_parts):
                        entry_points.append(str(file_path.relative_to(self.workspace_root)))
            else:
                # 精確匹配
                entry_file = module_path / entry_pattern
                if entry_file.exists():
                    entry_points.append(str(entry_file.relative_to(self.workspace_root)))
        
        return ModuleInfo(
            name=module_config["name"],
            path=str(module_path.relative_to(self.workspace_root)),
            language=primary_language,
            file_count=file_count,
            line_count=line_count,
            size_bytes=size_bytes,
            last_modified=last_modified,
            dependencies=dependencies,
            entry_points=entry_points,
            config_files=config_files
        )
    
    async def _detect_dependencies(self, module_path: Path, language: str) -> List[str]:
        """檢測模組依賴"""
        dependencies = []
        
        try:
            if language == "Python":
                # 檢查 requirements.txt
                req_file = module_path / "requirements.txt"
                if req_file.exists():
                    with open(req_file, 'r') as f:
                        dependencies.extend([line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')])
                
                # 檢查 Python 文件中的 import
                for py_file in module_path.rglob("*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # 簡單的 import 檢測
                            import re
                            imports = re.findall(r'^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content, re.MULTILINE)
                            dependencies.extend(imports)
                    except (UnicodeDecodeError, PermissionError):
                        continue
            
            elif language == "Rust":
                # 檢查 Cargo.toml
                cargo_file = module_path / "Cargo.toml"
                if cargo_file.exists():
                    try:
                        import toml
                        with open(cargo_file, 'r') as f:
                            cargo_data = toml.load(f)
                            if 'dependencies' in cargo_data:
                                dependencies.extend(cargo_data['dependencies'].keys())
                    except ImportError:
                        # 如果沒有 toml，使用簡單文本解析
                        with open(cargo_file, 'r') as f:
                            content = f.read()
                            # 簡單解析 [dependencies] 區塊
                            if '[dependencies]' in content:
                                deps_section = content.split('[dependencies]')[1].split('[')[0]
                                for line in deps_section.split('\n'):
                                    if '=' in line:
                                        dep_name = line.split('=')[0].strip()
                                        if dep_name:
                                            dependencies.append(dep_name)
            
            elif language == "Go":
                # 檢查 go.mod
                go_mod_file = module_path / "go.mod"
                if go_mod_file.exists():
                    with open(go_mod_file, 'r') as f:
                        content = f.read()
                        # 簡單解析 require 區塊
                        lines = content.split('\n')
                        in_require = False
                        for line in lines:
                            line = line.strip()
                            if line == 'require (':
                                in_require = True
                            elif line == ')' and in_require:
                                in_require = False
                            elif in_require and line:
                                dep_name = line.split()[0] if line.split() else ''
                                if dep_name:
                                    dependencies.append(dep_name)
        
        except Exception as e:
            logger.warning(f"檢測依賴失敗: {e}")
        
        # 去重並過濾標準庫
        unique_deps = list(set(dependencies))
        # 過濾 Python 標準庫和常見系統模組
        filtered_deps = [dep for dep in unique_deps if dep not in [
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing', 're', 'collections',
            'asyncio', 'logging', 'unittest', 'traceback', 'subprocess', 'platform'
        ]]
        
        return filtered_deps[:20]  # 限制數量避免過多
    
    async def _check_module_health(self, module_info: ModuleInfo, module_config: Dict[str, Any]) -> HealthStatus:
        """檢查模組健康狀態"""
        checks = {}
        issues = []
        warnings = []
        
        # 基本檢查
        checks["path_exists"] = Path(self.workspace_root / module_info.path).exists()
        if not checks["path_exists"]:
            issues.append(f"模組路徑不存在: {module_info.path}")
        
        checks["has_files"] = module_info.file_count > 0
        if not checks["has_files"]:
            issues.append("模組目錄為空")
        
        # 入口點檢查
        checks["has_entry_points"] = len(module_info.entry_points) > 0
        if not checks["has_entry_points"]:
            warnings.append("未找到入口點文件")
        
        # 依賴檢查
        missing_deps = await self._check_dependencies(module_info.dependencies, module_info.language)
        checks["dependencies_available"] = len(missing_deps) == 0
        if missing_deps:
            issues.extend([f"缺少依賴: {dep}" for dep in missing_deps])
        
        # 語言環境檢查
        lang_available = self.capabilities.get(module_info.language.lower(), False)
        checks["language_available"] = lang_available
        if not lang_available:
            issues.append(f"{module_info.language} 運行環境不可用")
        
        # 配置文件檢查
        checks["has_config"] = len(module_info.config_files) > 0
        if not checks["has_config"]:
            warnings.append("未找到配置文件")
        
        # 計算健康分數
        total_checks = len(checks)
        passed_checks = sum(checks.values())
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # 確定狀態
        if len(issues) > 0:
            status = "critical" if score < 0.5 else "warning"
        elif len(warnings) > 0:
            status = "warning"
        else:
            status = "healthy"
        
        return HealthStatus(
            status=status,
            score=score,
            checks=checks,
            issues=issues,
            warnings=warnings,
            last_check=datetime.now()
        )
    
    async def _check_dependencies(self, dependencies: List[str], language: str) -> List[str]:
        """檢查依賴是否可用"""
        missing = []
        
        for dep in dependencies:
            try:
                if language == "Python":
                    try:
                        __import__(dep)
                    except ImportError:
                        missing.append(dep)
                elif language == "Rust":
                    # Rust 依賴檢查較複雜，暫時跳過
                    pass
                elif language == "Go":
                    # Go 依賴檢查較複雜，暫時跳過
                    pass
            except Exception:
                continue
        
        return missing
    
    async def _collect_performance_metrics(self, module_info: ModuleInfo) -> PerformanceMetrics:
        """收集性能指標"""
        metrics = PerformanceMetrics()
        
        if self.capabilities.get("system_monitoring"):
            try:
                # 計算模組磁盤使用
                module_path = Path(self.workspace_root / module_info.path)
                if module_path.exists():
                    metrics.disk_usage = module_info.size_bytes / (1024 * 1024)  # MB
                
                # 如果是活躍模組，嘗試檢測運行時指標
                # 這裡可以擴展以檢測實際運行的程序
                
            except Exception as e:
                logger.warning(f"收集性能指標失敗: {e}")
        
        return metrics
    
    async def _check_communication_status(
        self, 
        module_info: ModuleInfo, 
        module_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """檢查模組間通信狀態"""
        status = {}
        
        # 檢查 RabbitMQ 連接（如果相關）
        if "pika" in module_info.dependencies or "rabbitmq" in str(module_info.path).lower():
            status["rabbitmq"] = "unknown"  # 實際實現中可以嘗試連接
        
        # 檢查 Redis 連接（如果相關）
        if "redis" in module_info.dependencies:
            status["redis"] = "unknown"
        
        # 檢查 HTTP 服務（如果相關）
        if any(dep in module_info.dependencies for dep in ["requests", "fastapi", "flask"]):
            status["http_client"] = "available"
        
        return status
    
    def _generate_module_recommendations(
        self,
        module_info: ModuleInfo,
        health_status: HealthStatus,
        performance_metrics: PerformanceMetrics,
        communication_status: Dict[str, str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """生成模組建議、CLI 指令和手動檢查項目"""
        recommendations = []
        cli_commands = []
        manual_checks = []
        
        # 基於健康狀態生成建議
        if health_status.status == "critical":
            recommendations.append("⚠️ 模組處於關鍵狀態，需要立即關注")
            if not health_status.checks.get("path_exists"):
                recommendations.append("檢查模組路徑配置")
                cli_commands.append(f"ls -la {module_info.path}")
                manual_checks.append(f"手動檢查目錄是否存在: {module_info.path}")
        
        # 依賴相關建議
        if health_status.issues:
            for issue in health_status.issues:
                if "缺少依賴" in issue:
                    dep_name = issue.split(": ")[1]
                    if module_info.language == "Python":
                        cli_commands.append(f"pip install {dep_name}")
                        manual_checks.append(f"手動安裝 Python 依賴: pip install {dep_name}")
                    elif module_info.language == "Rust":
                        cli_commands.append(f"cargo add {dep_name}")
                        manual_checks.append(f"手動添加 Rust 依賴到 Cargo.toml: {dep_name}")
                    elif module_info.language == "Go":
                        cli_commands.append(f"go get {dep_name}")
                        manual_checks.append(f"手動安裝 Go 依賴: go get {dep_name}")
        
        # 性能相關建議
        if performance_metrics.disk_usage and performance_metrics.disk_usage > 100:  # > 100MB
            recommendations.append("模組占用磁盤空間較大，考慮清理")
            cli_commands.append(f"du -sh {module_info.path}")
            manual_checks.append(f"手動檢查磁盤使用: du -sh {module_info.path}")
        
        # 通用檢查指令
        cli_commands.extend([
            f"# 檢查 {module_info.name} 狀態",
            f"find {module_info.path} -name '*.py' | wc -l",
            f"find {module_info.path} -name '*.log' -mtime -1"
        ])
        
        manual_checks.extend([
            f"檢查 {module_info.name} 的日誌文件",
            f"驗證 {module_info.name} 的配置文件",
            f"測試 {module_info.name} 的核心功能"
        ])
        
        return recommendations, cli_commands, manual_checks
    
    def _create_error_diagnostic(
        self, 
        module_id: str, 
        module_config: Dict[str, Any], 
        error_msg: str
    ) -> ModuleDiagnostic:
        """創建錯誤診斷結果"""
        
        module_info = ModuleInfo(
            name=module_config["name"],
            path=module_config["path"],
            language="Unknown",
            file_count=0,
            line_count=0,
            size_bytes=0,
            last_modified=datetime.min
        )
        
        health_status = HealthStatus(
            status="critical",
            score=0.0,
            issues=[f"探索失敗: {error_msg}"]
        )
        
        return ModuleDiagnostic(
            module_info=module_info,
            health_status=health_status,
            performance_metrics=PerformanceMetrics(),
            recommendations=[f"解決 {module_config['name']} 的探索錯誤"],
            cli_commands=[f"# 手動檢查 {module_config['name']}"],
            manual_checks=[f"人工檢查 {module_config['name']} 的狀態"]
        )
    
    def _calculate_overall_health(self, module_diagnostics: Dict[str, ModuleDiagnostic]) -> HealthStatus:
        """計算整體系統健康狀態"""
        if not module_diagnostics:
            return HealthStatus(status="unknown", score=0.0)
        
        scores = [diag.health_status.score for diag in module_diagnostics.values()]
        avg_score = sum(scores) / len(scores)
        
        critical_count = sum(1 for diag in module_diagnostics.values() 
                           if diag.health_status.status == "critical")
        warning_count = sum(1 for diag in module_diagnostics.values() 
                          if diag.health_status.status == "warning")
        
        all_issues = []
        all_warnings = []
        all_checks = {}
        
        for module_id, diag in module_diagnostics.items():
            all_issues.extend([f"{module_id}: {issue}" for issue in diag.health_status.issues])
            all_warnings.extend([f"{module_id}: {warning}" for warning in diag.health_status.warnings])
            for check, result in diag.health_status.checks.items():
                all_checks[f"{module_id}_{check}"] = result
        
        # 確定整體狀態
        if critical_count > len(module_diagnostics) / 2:
            status = "critical"
        elif critical_count > 0 or warning_count > len(module_diagnostics) / 2:
            status = "warning"  
        else:
            status = "healthy"
        
        return HealthStatus(
            status=status,
            score=avg_score,
            checks=all_checks,
            issues=all_issues,
            warnings=all_warnings,
            last_check=datetime.now()
        )
    
    def _generate_system_recommendations(
        self,
        module_diagnostics: Dict[str, ModuleDiagnostic],
        overall_health: HealthStatus
    ) -> Tuple[List[str], List[str], List[str]]:
        """生成系統級建議"""
        critical_issues = []
        optimization_suggestions = []
        next_actions = []
        
        # 收集關鍵問題
        for module_id, diag in module_diagnostics.items():
            if diag.health_status.status == "critical":
                critical_issues.append(f"{diag.module_info.name}: {diag.health_status.status}")
                for issue in diag.health_status.issues[:2]:  # 限制數量
                    critical_issues.append(f"  - {issue}")
        
        # 生成優化建議
        healthy_modules = [diag for diag in module_diagnostics.values() 
                         if diag.health_status.status == "healthy"]
        if len(healthy_modules) > 3:
            optimization_suggestions.append("大部分模組健康，可考慮性能優化")
        
        total_lines = sum(diag.module_info.line_count for diag in module_diagnostics.values())
        if total_lines > 100000:
            optimization_suggestions.append("程式碼規模較大，建議進行模組化重構")
        
        # 生成下一步行動
        if overall_health.status == "critical":
            next_actions.append("立即修復關鍵問題")
            next_actions.append("檢查系統依賴和配置")
        elif overall_health.status == "warning":
            next_actions.append("解決警告問題")
            next_actions.append("優化模組性能")
        else:
            next_actions.append("保持系統健康狀態")
            next_actions.append("考慮新功能開發")
        
        # 添加 CLI 相關行動
        next_actions.append("使用生成的 CLI 指令進行深入檢查")
        next_actions.append("執行手動檢查項目確認狀態")
        
        return critical_issues, optimization_suggestions, next_actions
    
    async def _save_diagnostic_report(self, report: SystemDiagnosticReport) -> None:
        """保存診斷報告"""
        try:
            # 確保報告目錄存在
            reports_dir = self.workspace_root / "reports" / "ai_diagnostics"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成報告文件名
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"system_diagnostic_{timestamp}_{report.report_id}.json"
            
            # 轉換為 JSON 可序列化格式
            report_dict = self._serialize_report(report)
            
            # 保存 JSON 報告
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
            
            # 生成簡要版本用於快速查看
            summary_file = reports_dir / f"summary_{timestamp}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_summary_text(report))
            
            logger.info(f"📄 診斷報告已保存: {report_file}")
            logger.info(f"📋 報告摘要已保存: {summary_file}")
            
        except Exception as e:
            logger.error(f"保存診斷報告失敗: {e}")
    
    def _serialize_report(self, report: SystemDiagnosticReport) -> Dict[str, Any]:
        """序列化報告為 JSON 格式"""
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "system_info": report.system_info,
            "modules": {
                module_id: {
                    "module_info": asdict(diag.module_info),
                    "health_status": asdict(diag.health_status),
                    "performance_metrics": asdict(diag.performance_metrics),
                    "communication_status": diag.communication_status,
                    "recommendations": diag.recommendations,
                    "cli_commands": diag.cli_commands,
                    "manual_checks": diag.manual_checks
                }
                for module_id, diag in report.modules.items()
            },
            "overall_health": asdict(report.overall_health),
            "critical_issues": report.critical_issues,
            "optimization_suggestions": report.optimization_suggestions,
            "next_actions": report.next_actions
        }
    
    def _generate_summary_text(self, report: SystemDiagnosticReport) -> str:
        """生成文本摘要"""
        lines = [
            f"AIVA 系統診斷報告摘要",
            f"=" * 50,
            f"報告 ID: {report.report_id}",
            f"生成時間: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"整體健康狀態: {report.overall_health.status.upper()} (分數: {report.overall_health.score:.2f})",
            "",
            "模組狀態概覽:",
            "-" * 30
        ]
        
        # 模組狀態
        for module_id, diag in report.modules.items():
            status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(
                diag.health_status.status, "❓"
            )
            lines.append(
                f"{status_icon} {diag.module_info.name}: {diag.health_status.status} "
                f"({diag.module_info.file_count} 文件, {diag.module_info.line_count} 行)"
            )
        
        # 關鍵問題
        if report.critical_issues:
            lines.extend([
                "",
                "⚠️ 關鍵問題:",
                "-" * 15
            ])
            lines.extend([f"  • {issue}" for issue in report.critical_issues[:5]])
        
        # 建議行動
        if report.next_actions:
            lines.extend([
                "",
                "🎯 建議行動:",
                "-" * 15
            ])
            lines.extend([f"  • {action}" for action in report.next_actions[:5]])
        
        # CLI 指令示例
        lines.extend([
            "",
            "🖥️ 可用 CLI 指令 (示例):",
            "-" * 25
        ])
        
        cli_count = 0
        for diag in report.modules.values():
            for cmd in diag.cli_commands[:2]:  # 每個模組最多2個指令
                if not cmd.startswith("#"):
                    lines.append(f"  {cmd}")
                    cli_count += 1
                    if cli_count >= 5:
                        break
            if cli_count >= 5:
                break
        
        lines.extend([
            "",
            "📋 手動檢查項目:",
            "-" * 18
        ])
        
        manual_count = 0
        for diag in report.modules.values():
            for check in diag.manual_checks[:2]:
                lines.append(f"  • {check}")
                manual_count += 1
                if manual_count >= 5:
                    break
            if manual_count >= 5:
                break
        
        lines.extend([
            "",
            f"完整報告請查看 JSON 文件: system_diagnostic_{report.timestamp.strftime('%Y%m%d_%H%M%S')}_{report.report_id}.json"
        ])
        
        return "\n".join(lines)


# CLI 介面
async def main():
    """命令行介面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA AI 系統自我探索器")
    parser.add_argument("--workspace", "-w", type=str, help="工作目錄路徑")
    parser.add_argument("--detailed", "-d", action="store_true", help="詳細模式")
    parser.add_argument("--module", "-m", type=str, help="指定探索的模組 (ai_core, attack_engine, scan_engine, integration_service, feature_detection)")
    parser.add_argument("--output", "-o", type=str, help="輸出格式 (json, text, both)", default="both")
    
    args = parser.parse_args()
    
    # 初始化探索器
    workspace = Path(args.workspace) if args.workspace else Path.cwd()
    explorer = SystemSelfExplorer(workspace)
    
    print(f"🔍 AIVA 系統自我探索器 v1.0.0")
    print(f"📁 工作目錄: {workspace}")
    print(f"🎯 探索模式: {'詳細' if args.detailed else '標準'}")
    
    if args.module:
        if args.module not in explorer.CORE_MODULES:
            print(f"❌ 未知模組: {args.module}")
            print(f"可用模組: {', '.join(explorer.CORE_MODULES.keys())}")
            return
        
        print(f"🎯 指定模組: {args.module}")
        # 單模組探索
        module_config = explorer.CORE_MODULES[args.module]
        diagnostic = await explorer._explore_module(args.module, module_config, args.detailed)
        
        print(f"\n📊 {diagnostic.module_info.name} 診斷結果:")
        print(f"健康狀態: {diagnostic.health_status.status} (分數: {diagnostic.health_status.score:.2f})")
        print(f"檔案數量: {diagnostic.module_info.file_count}")
        print(f"程式碼行數: {diagnostic.module_info.line_count}")
        
        if diagnostic.health_status.issues:
            print(f"\n⚠️ 發現問題:")
            for issue in diagnostic.health_status.issues:
                print(f"  • {issue}")
        
        if diagnostic.recommendations:
            print(f"\n💡 建議:")
            for rec in diagnostic.recommendations:
                print(f"  • {rec}")
    
    else:
        # 完整系統探索
        report = await explorer.explore_system(args.detailed)
        
        print(f"\n📊 系統診斷完成!")
        print(f"整體健康狀態: {report.overall_health.status.upper()} (分數: {report.overall_health.score:.2f})")
        print(f"探索模組數量: {len(report.modules)}")
        
        if report.critical_issues:
            print(f"\n⚠️ 關鍵問題:")
            for issue in report.critical_issues[:3]:
                print(f"  • {issue}")
        
        if report.next_actions:
            print(f"\n🎯 建議行動:")
            for action in report.next_actions[:3]:
                print(f"  • {action}")
        
        print(f"\n📄 詳細報告已保存到 reports/ai_diagnostics/ 目錄")


if __name__ == "__main__":
    asyncio.run(main())