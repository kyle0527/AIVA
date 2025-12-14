from __future__ import annotations
import asyncio, importlib, inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

# Import native detection models
from ..detection_models import DetectionResult

class SqliEngineProtocol(Protocol):
    async def detect(self, target: str, params: Dict[str, Any]) -> List[DetectionResult]: ...  # noqa: E701

_ENGINE_CANDIDATES = [
    ("..engines.boolean_detection_engine", "BooleanDetectionEngine"),
    ("..engines.time_detection_engine", "TimeDetectionEngine"),
    ("..engines.union_detection_engine", "UnionDetectionEngine"),
    ("..engines.error_detection_engine", "ErrorDetectionEngine"),
    ("..engines.oob_detection_engine", "OOBDetectionEngine"),
    ("..engines.hackingtool_engine", "HackingToolDetectionEngine"),
]

class SqliDetector:
    """Unified SQLi detector that orchestrates multiple engines.
    - Dynamic import: only available engines are loaded.
    - Parallel execution: asyncio.gather for speed.
    - Smart selection (optional): you can pass 'db_fingerprint' via params to bias engines.
    """
    def __init__(self) -> None:
        self.engines: List[SqliEngineProtocol] = []
        for mod, cls in _ENGINE_CANDIDATES:
            eng = self._try_import_engine(mod, cls)
            if eng:
                self.engines.append(eng)

    def _try_import_engine(self, mod_name: str, cls_name: str) -> Optional[SqliEngineProtocol]:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                return None
            inst = cls()  # type: ignore[call-arg]
            # if engine.detect is sync, wrap as async
            if not inspect.iscoroutinefunction(getattr(inst, "detect", None)):
                sync_detect = inst.detect  # type: ignore[attr-defined]
                async def _async_wrapper(target: str, params: Dict[str, Any]) -> List[DetectionResult]:
                    return await asyncio.to_thread(sync_detect, target, params)
                inst.detect = _async_wrapper  # type: ignore[assignment]
            return inst  # type: ignore[return-value]
        except Exception:
            return None

    async def detect_sqli(self, target: str, params: Dict[str, Any]) -> List[DetectionResult]:
        """執行SQL注入檢測"""
        if not self.engines:
            return []
        
        # 根據資料庫指紋排序引擎
        dbfp = params.get("db_fingerprint") or params.get("db_type")
        ordered_engines = self._order_engines(dbfp)
        
        # 並行執行檢測
        results = await self._execute_parallel_detection(target, params, ordered_engines)
        
        # 處理和合併結果
        return self._process_and_merge_results(results)
    
    async def _execute_parallel_detection(self, target: str, params: Dict[str, Any], engines: List[SqliEngineProtocol]) -> List[List[DetectionResult]]:
        """並行執行檢測"""
        tasks = [engine.detect(target, params) for engine in engines]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results_nested if isinstance(r, list)]
    
    def _process_and_merge_results(self, results_nested: List[List[DetectionResult]]) -> List[DetectionResult]:
        """處理和合併結果"""
        # 展平結果
        flat_results = []
        for result_list in results_nested:
            flat_results.extend(result_list)
        
        # 去重和標準化
        return self._deduplicate_and_normalize(flat_results)
    
    def _deduplicate_and_normalize(self, results: List[DetectionResult]) -> List[DetectionResult]:
        """去重和標準化結果"""
        seen = set()
        merged = []
        
        for dr in results:
            # 使用新的 DetectionResult 結構
            key = (dr.detection_method, dr.payload_used, dr.target.parameter if dr.target.parameter else "")
            if key in seen:
                continue
            seen.add(key)
            
            # DetectionResult 使用 Vulnerability 對象，不需要手動標準化
            # 嚴重度和置信度已在創建時設置
            
            merged.append(dr)
        
        return merged

    # basic ordering heuristic
    def _order_engines(self, dbfp: Optional[str]) -> List[SqliEngineProtocol]:
        if not dbfp:
            return self.engines
        name = str(dbfp).lower()
        order: List[str] = []
        if any(k in name for k in ["mysql", "maria"]):
            order = ["UnionDetectionEngine","BooleanDetectionEngine","ErrorDetectionEngine","TimeDetectionEngine","OOBDetectionEngine","HackingToolDetectionEngine"]
        elif "postgres" in name or "psql" in name:
            order = ["BooleanDetectionEngine","TimeDetectionEngine","UnionDetectionEngine","ErrorDetectionEngine","OOBDetectionEngine","HackingToolDetectionEngine"]
        elif any(k in name for k in ["mssql","sqlserver"]):
            order = ["ErrorDetectionEngine","UnionDetectionEngine","BooleanDetectionEngine","TimeDetectionEngine","OOBDetectionEngine","HackingToolDetectionEngine"]
        elif "oracle" in name:
            order = ["UnionDetectionEngine","ErrorDetectionEngine","BooleanDetectionEngine","TimeDetectionEngine","OOBDetectionEngine","HackingToolDetectionEngine"]
        else:
            return self.engines
        # re-order by matching class names
        def idx(e): 
            n = e.__class__.__name__
            return order.index(n) if n in order else 999
        return sorted(self.engines, key=idx)
