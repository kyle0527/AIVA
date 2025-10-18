#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA 跨語言方案智能選擇器
根據環境可用性、性能需求、安全性等因素自動選擇最佳通信方法
"""

import os
import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """性能等級"""
    LOW = "low"           # 低性能需求
    MEDIUM = "medium"     # 中等性能需求
    HIGH = "high"         # 高性能需求
    CRITICAL = "critical" # 關鍵性能需求

class SecurityLevel(Enum):
    """安全等級"""
    BASIC = "basic"       # 基本安全
    STANDARD = "standard" # 標準安全
    HIGH = "high"         # 高安全性
    CRITICAL = "critical" # 關鍵安全性

class ReliabilityLevel(Enum):
    """可靠性等級"""
    BASIC = "basic"       # 基本可靠性
    STANDARD = "standard" # 標準可靠性
    HIGH = "high"         # 高可靠性
    CRITICAL = "critical" # 關鍵可靠性

@dataclass
class CommunicationRequirement:
    """通信需求規格"""
    performance: PerformanceLevel
    security: SecurityLevel
    reliability: ReliabilityLevel
    data_size: str = "small"  # small, medium, large, huge
    latency_sensitive: bool = False
    bidirectional: bool = True
    persistent_connection: bool = False
    cross_network: bool = False
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["python"]

@dataclass
class CommunicationMethod:
    """通信方法規格"""
    name: str
    type: str  # ffi, ipc, network, memory, file
    performance_score: float  # 0.0 - 1.0
    security_score: float     # 0.0 - 1.0
    reliability_score: float  # 0.0 - 1.0
    setup_complexity: float   # 0.0 - 1.0 (lower is better)
    resource_usage: float     # 0.0 - 1.0 (lower is better)
    supported_languages: List[str]
    max_data_size: str       # small, medium, large, huge
    latency: str             # ultra_low, low, medium, high
    requires_external_deps: bool
    cross_network_capable: bool
    bidirectional_support: bool
    persistent_connection: bool
    availability_check: Optional[callable] = None

class CrossLanguageSelector:
    """跨語言方案選擇器"""
    
    def __init__(self):
        self.methods = self._initialize_methods()
        self.availability_cache = {}
        self.performance_cache = {}
        self.logger = logging.getLogger("CrossLanguageSelector")
    
    def _initialize_methods(self) -> Dict[str, CommunicationMethod]:
        """初始化通信方法"""
        methods = {}
        
        # FFI 方法
        methods["rust_ffi"] = CommunicationMethod(
            name="Rust FFI",
            type="ffi",
            performance_score=0.95,
            security_score=0.85,
            reliability_score=0.90,
            setup_complexity=0.7,
            resource_usage=0.1,
            supported_languages=["python", "rust"],
            max_data_size="large",
            latency="ultra_low",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_rust_ffi_availability
        )
        
        methods["go_ffi"] = CommunicationMethod(
            name="Go FFI",
            type="ffi",
            performance_score=0.90,
            security_score=0.80,
            reliability_score=0.85,
            setup_complexity=0.6,
            resource_usage=0.15,
            supported_languages=["python", "go"],
            max_data_size="large",
            latency="ultra_low",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_go_ffi_availability
        )
        
        methods["cffi"] = CommunicationMethod(
            name="CFFI",
            type="ffi",
            performance_score=0.85,
            security_score=0.75,
            reliability_score=0.80,
            setup_complexity=0.5,
            resource_usage=0.1,
            supported_languages=["python", "c", "cpp"],
            max_data_size="medium",
            latency="ultra_low",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_cffi_availability
        )
        
        # WebAssembly 方法
        methods["wasm_wasmtime"] = CommunicationMethod(
            name="WebAssembly (Wasmtime)",
            type="wasm",
            performance_score=0.75,
            security_score=0.95,
            reliability_score=0.85,
            setup_complexity=0.8,
            resource_usage=0.3,
            supported_languages=["python", "rust", "c", "cpp"],
            max_data_size="medium",
            latency="low",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_wasmtime_availability
        )
        
        methods["wasm_wasmer"] = CommunicationMethod(
            name="WebAssembly (Wasmer)",
            type="wasm",
            performance_score=0.70,
            security_score=0.95,
            reliability_score=0.80,
            setup_complexity=0.8,
            resource_usage=0.35,
            supported_languages=["python", "rust", "c", "cpp"],
            max_data_size="medium",
            latency="low",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_wasmer_availability
        )
        
        # GraalVM 方法
        methods["graalvm"] = CommunicationMethod(
            name="GraalVM Polyglot",
            type="polyglot",
            performance_score=0.80,
            security_score=0.90,
            reliability_score=0.85,
            setup_complexity=0.9,
            resource_usage=0.5,
            supported_languages=["python", "javascript", "java", "ruby", "r"],
            max_data_size="large",
            latency="low",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_graalvm_availability
        )
        
        methods["nodejs_fallback"] = CommunicationMethod(
            name="Node.js Fallback",
            type="subprocess",
            performance_score=0.50,
            security_score=0.60,
            reliability_score=0.70,
            setup_complexity=0.3,
            resource_usage=0.4,
            supported_languages=["python", "javascript"],
            max_data_size="medium",
            latency="medium",
            requires_external_deps=True,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_nodejs_availability
        )
        
        # IPC 方法
        methods["tcp_socket"] = CommunicationMethod(
            name="TCP Socket",
            type="network",
            performance_score=0.70,
            security_score=0.50,
            reliability_score=0.85,
            setup_complexity=0.4,
            resource_usage=0.2,
            supported_languages=["python", "go", "rust", "javascript", "java", "c", "cpp"],
            max_data_size="huge",
            latency="medium",
            requires_external_deps=False,
            cross_network_capable=True,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_tcp_availability
        )
        
        methods["websocket"] = CommunicationMethod(
            name="WebSocket",
            type="network",
            performance_score=0.65,
            security_score=0.70,
            reliability_score=0.80,
            setup_complexity=0.5,
            resource_usage=0.25,
            supported_languages=["python", "javascript", "go", "rust", "java"],
            max_data_size="large",
            latency="medium",
            requires_external_deps=True,
            cross_network_capable=True,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_websocket_availability
        )
        
        methods["zmq"] = CommunicationMethod(
            name="ZeroMQ",
            type="network",
            performance_score=0.85,
            security_score=0.75,
            reliability_score=0.90,
            setup_complexity=0.6,
            resource_usage=0.15,
            supported_languages=["python", "go", "rust", "c", "cpp", "java"],
            max_data_size="huge",
            latency="low",
            requires_external_deps=True,
            cross_network_capable=True,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_zmq_availability
        )
        
        methods["grpc"] = CommunicationMethod(
            name="gRPC",
            type="network",
            performance_score=0.80,
            security_score=0.85,
            reliability_score=0.90,
            setup_complexity=0.8,
            resource_usage=0.3,
            supported_languages=["python", "go", "rust", "javascript", "java", "c", "cpp"],
            max_data_size="huge",
            latency="low",
            requires_external_deps=True,
            cross_network_capable=True,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_grpc_availability
        )
        
        # 記憶體共享方法
        methods["shared_memory"] = CommunicationMethod(
            name="Shared Memory",
            type="memory",
            performance_score=0.95,
            security_score=0.60,
            reliability_score=0.75,
            setup_complexity=0.7,
            resource_usage=0.05,
            supported_languages=["python", "c", "cpp", "rust", "go"],
            max_data_size="huge",
            latency="ultra_low",
            requires_external_deps=False,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_shared_memory_availability
        )
        
        methods["named_pipe"] = CommunicationMethod(
            name="Named Pipe",
            type="ipc",
            performance_score=0.75,
            security_score=0.70,
            reliability_score=0.80,
            setup_complexity=0.4,
            resource_usage=0.1,
            supported_languages=["python", "c", "cpp", "go", "rust"],
            max_data_size="large",
            latency="low",
            requires_external_deps=False,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=True,
            availability_check=self._check_named_pipe_availability
        )
        
        # 檔案系統方法
        methods["file_based"] = CommunicationMethod(
            name="File-based Communication",
            type="file",
            performance_score=0.30,
            security_score=0.50,
            reliability_score=0.95,
            setup_complexity=0.1,
            resource_usage=0.05,
            supported_languages=["python", "go", "rust", "javascript", "java", "c", "cpp"],
            max_data_size="huge",
            latency="high",
            requires_external_deps=False,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_file_based_availability
        )
        
        # 子程序方法
        methods["subprocess"] = CommunicationMethod(
            name="Subprocess",
            type="subprocess",
            performance_score=0.40,
            security_score=0.40,
            reliability_score=0.70,
            setup_complexity=0.2,
            resource_usage=0.3,
            supported_languages=["python", "go", "rust", "javascript", "java", "c", "cpp"],
            max_data_size="large",
            latency="medium",
            requires_external_deps=False,
            cross_network_capable=False,
            bidirectional_support=True,
            persistent_connection=False,
            availability_check=self._check_subprocess_availability
        )
        
        return methods
    
    async def select_best_method(self, requirement: CommunicationRequirement) -> Tuple[str, CommunicationMethod, float]:
        """選擇最佳通信方法"""
        self.logger.info(f"選擇最佳通信方法，需求: {requirement}")
        
        # 獲取可用方法
        available_methods = await self._get_available_methods()
        
        if not available_methods:
            raise RuntimeError("沒有可用的通信方法")
        
        # 計算每個方法的得分
        method_scores = []
        
        for method_name, method in available_methods.items():
            score = await self._calculate_method_score(method, requirement)
            method_scores.append((method_name, method, score))
            
            self.logger.info(f"{method.name}: {score:.3f}")
        
        # 排序並選擇最佳方法
        method_scores.sort(key=lambda x: x[2], reverse=True)
        
        best_method = method_scores[0]
        self.logger.info(f"選擇最佳方法: {best_method[1].name} (得分: {best_method[2]:.3f})")
        
        return best_method
    
    async def get_fallback_methods(self, requirement: CommunicationRequirement, 
                                 exclude: List[str] = None) -> List[Tuple[str, CommunicationMethod, float]]:
        """獲取備用方法列表"""
        exclude = exclude or []
        
        # 獲取可用方法
        available_methods = await self._get_available_methods()
        
        # 移除排除的方法
        for method_name in exclude:
            available_methods.pop(method_name, None)
        
        # 計算得分
        method_scores = []
        
        for method_name, method in available_methods.items():
            score = await self._calculate_method_score(method, requirement)
            method_scores.append((method_name, method, score))
        
        # 排序
        method_scores.sort(key=lambda x: x[2], reverse=True)
        
        return method_scores
    
    async def _get_available_methods(self) -> Dict[str, CommunicationMethod]:
        """獲取可用的通信方法"""
        available = {}
        
        for method_name, method in self.methods.items():
            if method_name in self.availability_cache:
                is_available = self.availability_cache[method_name]
            else:
                is_available = await self._check_method_availability(method)
                self.availability_cache[method_name] = is_available
            
            if is_available:
                available[method_name] = method
        
        return available
    
    async def _check_method_availability(self, method: CommunicationMethod) -> bool:
        """檢查方法可用性"""
        try:
            if method.availability_check:
                return await method.availability_check()
            else:
                return True  # 如果沒有檢查函數，假設可用
        except Exception as e:
            self.logger.warning(f"檢查 {method.name} 可用性失敗: {e}")
            return False
    
    async def _calculate_method_score(self, method: CommunicationMethod, 
                                    requirement: CommunicationRequirement) -> float:
        """計算方法得分"""
        score = 0.0
        
        # 性能得分 (30%)
        perf_weight = 0.3
        if requirement.performance == PerformanceLevel.CRITICAL:
            score += method.performance_score * perf_weight
        elif requirement.performance == PerformanceLevel.HIGH:
            score += method.performance_score * perf_weight * 0.8
        elif requirement.performance == PerformanceLevel.MEDIUM:
            score += method.performance_score * perf_weight * 0.6
        else:  # LOW
            score += method.performance_score * perf_weight * 0.4
        
        # 安全得分 (25%)
        sec_weight = 0.25
        if requirement.security == SecurityLevel.CRITICAL:
            score += method.security_score * sec_weight
        elif requirement.security == SecurityLevel.HIGH:
            score += method.security_score * sec_weight * 0.8
        elif requirement.security == SecurityLevel.STANDARD:
            score += method.security_score * sec_weight * 0.6
        else:  # BASIC
            score += method.security_score * sec_weight * 0.4
        
        # 可靠性得分 (25%)
        rel_weight = 0.25
        if requirement.reliability == ReliabilityLevel.CRITICAL:
            score += method.reliability_score * rel_weight
        elif requirement.reliability == ReliabilityLevel.HIGH:
            score += method.reliability_score * rel_weight * 0.8
        elif requirement.reliability == ReliabilityLevel.STANDARD:
            score += method.reliability_score * rel_weight * 0.6
        else:  # BASIC
            score += method.reliability_score * rel_weight * 0.4
        
        # 設定複雜度懲罰 (10%)
        complexity_weight = 0.1
        score += (1.0 - method.setup_complexity) * complexity_weight
        
        # 資源使用懲罰 (10%)
        resource_weight = 0.1
        score += (1.0 - method.resource_usage) * resource_weight
        
        # 語言支援加分
        supported_langs = set(method.supported_languages)
        required_langs = set(requirement.languages)
        
        if required_langs.issubset(supported_langs):
            score += 0.1  # 支援所有需要的語言
        else:
            score -= 0.2  # 不支援某些語言，大幅扣分
        
        # 特殊需求檢查
        if requirement.cross_network and not method.cross_network_capable:
            score -= 0.3  # 需要跨網路但不支援
        
        if requirement.persistent_connection and not method.persistent_connection:
            score -= 0.1  # 需要持久連接但不支援
        
        if requirement.bidirectional and not method.bidirectional_support:
            score -= 0.2  # 需要雙向通信但不支援
        
        # 延遲敏感性
        if requirement.latency_sensitive:
            if method.latency in ["ultra_low", "low"]:
                score += 0.1
            elif method.latency == "high":
                score -= 0.2
        
        # 資料大小匹配
        data_size_scores = {
            "small": {"small": 1.0, "medium": 0.8, "large": 0.6, "huge": 0.4},
            "medium": {"small": 0.6, "medium": 1.0, "large": 0.8, "huge": 0.6},
            "large": {"small": 0.4, "medium": 0.6, "large": 1.0, "huge": 0.8},
            "huge": {"small": 0.2, "medium": 0.4, "large": 0.6, "huge": 1.0}
        }
        
        size_match = data_size_scores.get(requirement.data_size, {}).get(method.max_data_size, 0.5)
        score += size_match * 0.05
        
        # 確保得分在 0-1 範圍內
        return max(0.0, min(1.0, score))
    
    # 可用性檢查函數
    async def _check_rust_ffi_availability(self) -> bool:
        """檢查 Rust FFI 可用性"""
        try:
            import subprocess
            result = subprocess.run(["cargo", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def _check_go_ffi_availability(self) -> bool:
        """檢查 Go FFI 可用性"""
        try:
            import subprocess
            result = subprocess.run(["go", "version"], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def _check_cffi_availability(self) -> bool:
        """檢查 CFFI 可用性"""
        try:
            import cffi
            return True
        except ImportError:
            return False
    
    async def _check_wasmtime_availability(self) -> bool:
        """檢查 Wasmtime 可用性"""
        try:
            import wasmtime
            return True
        except ImportError:
            return False
    
    async def _check_wasmer_availability(self) -> bool:
        """檢查 Wasmer 可用性"""
        try:
            import wasmer
            return True
        except ImportError:
            return False
    
    async def _check_graalvm_availability(self) -> bool:
        """檢查 GraalVM 可用性"""
        try:
            import polyglot
            return True
        except ImportError:
            return False
    
    async def _check_nodejs_availability(self) -> bool:
        """檢查 Node.js 可用性"""
        try:
            import subprocess
            result = subprocess.run(["node", "--version"], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def _check_tcp_availability(self) -> bool:
        """檢查 TCP Socket 可用性"""
        return True  # Python 內建支援
    
    async def _check_websocket_availability(self) -> bool:
        """檢查 WebSocket 可用性"""
        try:
            import websockets
            return True
        except ImportError:
            return False
    
    async def _check_zmq_availability(self) -> bool:
        """檢查 ZeroMQ 可用性"""
        try:
            import zmq
            return True
        except ImportError:
            return False
    
    async def _check_grpc_availability(self) -> bool:
        """檢查 gRPC 可用性"""
        try:
            import grpc
            return True
        except ImportError:
            return False
    
    async def _check_shared_memory_availability(self) -> bool:
        """檢查共享記憶體可用性"""
        try:
            import mmap
            return True
        except ImportError:
            return False
    
    async def _check_named_pipe_availability(self) -> bool:
        """檢查具名管道可用性"""
        return True  # Python 內建支援
    
    async def _check_file_based_availability(self) -> bool:
        """檢查檔案系統通信可用性"""
        return True  # Python 內建支援
    
    async def _check_subprocess_availability(self) -> bool:
        """檢查子程序可用性"""
        return True  # Python 內建支援

class AIVASmartCommunicationManager:
    """AIVA 智能通信管理器"""
    
    def __init__(self):
        self.selector = CrossLanguageSelector()
        self.active_connections = {}
        self.logger = logging.getLogger("AIVASmartCommunicationManager")
    
    async def get_optimal_communication_setup(self, requirement: CommunicationRequirement) -> Dict[str, Any]:
        """獲取最佳通信設定"""
        # 選擇主要方法
        primary_method = await self.selector.select_best_method(requirement)
        
        # 獲取備用方法
        fallback_methods = await self.selector.get_fallback_methods(
            requirement, exclude=[primary_method[0]]
        )
        
        # 建立配置
        config = {
            "primary": {
                "name": primary_method[0],
                "method": primary_method[1],
                "score": primary_method[2]
            },
            "fallbacks": [
                {
                    "name": method[0],
                    "method": method[1],
                    "score": method[2]
                }
                for method in fallback_methods[:3]  # 最多3個備用方案
            ],
            "requirement": requirement,
            "recommendation": self._generate_setup_recommendation(primary_method, fallback_methods)
        }
        
        return config
    
    def _generate_setup_recommendation(self, primary, fallbacks) -> str:
        """生成設定建議"""
        primary_method = primary[1]
        
        recommendations = [
            f"建議使用 {primary_method.name} 作為主要通信方法"
        ]
        
        if primary_method.requires_external_deps:
            recommendations.append(f"需要安裝外部依賴")
        
        if primary_method.setup_complexity > 0.7:
            recommendations.append("設定較為複雜，建議準備備用方案")
        
        if fallbacks:
            fallback_names = [f[1].name for f in fallbacks[:2]]
            recommendations.append(f"備用方案: {', '.join(fallback_names)}")
        
        return "; ".join(recommendations)

# 使用範例和測試
async def demo_smart_selection():
    """示範智能選擇功能"""
    manager = AIVASmartCommunicationManager()
    
    # 測試案例 1: 高性能、低延遲需求
    print("🔍 測試案例 1: 高性能安全掃描")
    requirement1 = CommunicationRequirement(
        performance=PerformanceLevel.HIGH,
        security=SecurityLevel.HIGH,
        reliability=ReliabilityLevel.STANDARD,
        data_size="medium",
        latency_sensitive=True,
        languages=["python", "rust"]
    )
    
    config1 = await manager.get_optimal_communication_setup(requirement1)
    print(f"主要方法: {config1['primary']['method'].name} (得分: {config1['primary']['score']:.3f})")
    print(f"建議: {config1['recommendation']}")
    
    # 測試案例 2: 跨網路、大資料傳輸
    print("\n🔍 測試案例 2: 跨網路大資料處理")
    requirement2 = CommunicationRequirement(
        performance=PerformanceLevel.MEDIUM,
        security=SecurityLevel.STANDARD,
        reliability=ReliabilityLevel.HIGH,
        data_size="huge",
        cross_network=True,
        persistent_connection=True,
        languages=["python", "go", "javascript"]
    )
    
    config2 = await manager.get_optimal_communication_setup(requirement2)
    print(f"主要方法: {config2['primary']['method'].name} (得分: {config2['primary']['score']:.3f})")
    print(f"建議: {config2['recommendation']}")
    
    # 測試案例 3: 簡單可靠的通信
    print("\n🔍 測試案例 3: 簡單可靠通信")
    requirement3 = CommunicationRequirement(
        performance=PerformanceLevel.LOW,
        security=SecurityLevel.BASIC,
        reliability=ReliabilityLevel.HIGH,
        data_size="small",
        languages=["python"]
    )
    
    config3 = await manager.get_optimal_communication_setup(requirement3)
    print(f"主要方法: {config3['primary']['method'].name} (得分: {config3['primary']['score']:.3f})")
    print(f"建議: {config3['recommendation']}")
    
    # 顯示所有可用方法
    print("\n📋 所有可用的通信方法:")
    available = await manager.selector._get_available_methods()
    for name, method in available.items():
        print(f"  - {method.name} ({method.type})")

if __name__ == "__main__":
    print("🧠 AIVA 智能跨語言通信選擇器")
    print("=" * 50)
    
    # 執行示範
    asyncio.run(demo_smart_selection())