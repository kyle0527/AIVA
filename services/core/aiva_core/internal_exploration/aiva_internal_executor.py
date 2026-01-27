#!/usr/bin/env python3
"""
AIVA CLI Implementation - Dynamic Flow Executor & Documentation Generator
動態流程執行器與指令手冊生成工具

這個模組整合了兩大功能:
1. **動態流程執行 (Flow Execution)**: 
   - 讀取 classification_data.json 中的數據流定義
   - 動態導入模組、實例化類別、自動偵測並執行入口方法
   - 支援步驟間的 Pipeline 數據傳遞
   - 提供 Dry Run 模式預覽執行計畫

2. **指令手冊生成 (Documentation Generation)**:
   - 生成 Markdown 格式的 CLI 指令參考手冊 (適合人類閱讀)
   - 生成 JSON 格式的指令資料庫 (適合 AI 檢索)

核心特性:
- 自動處理 Windows 絕對路徑與 Python 模組路徑轉換
- 智能類別名稱推斷 (snake_case → CamelCase)
- 啟發式入口方法偵測 (train, execute, run, process, analyze 等)
- 容錯機制: 找不到類別時自動搜尋模組內其他類別
- Pipeline 執行: 自動在步驟間傳遞輸出數據

使用方式:
    # 生成 Markdown 參考手冊
    python -m aiva_core.internal_exploration.aiva_cli_implementation --generate-doc md
    
    # 列出可用流程
    python -m aiva_core.internal_exploration.aiva_cli_implementation --list
    
    # 預覽執行計畫 (不實際執行)
    python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11 --dry-run
    
    # 實際執行流程
    python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11

依賴:
- classification_data.json: 包含所有數據流的定義 (282 flows)

輸出:
- 直接執行數據流,在步驟間傳遞數據
- 生成 CLI_COMMANDS_REFERENCE.md (Markdown 手冊)
- 生成 cli_commands_db.json (JSON 資料庫)
"""

import sys
import os
import json
import importlib
import argparse
import inspect
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from glob import glob

# ==========================================
# 設定與常數
# ==========================================

# 設定預設的 JSON 資料來源路徑
# 優先使用 latest_classification.json (由 pipeline 自動更新)
# 後備方案使用 services_classification_v9_new/classification_data.json
CURRENT_DIR = Path(__file__).resolve().parent
LATEST_DATA_PATH = CURRENT_DIR / "latest_classification.json"
DEFAULT_JSON_PATH = CURRENT_DIR / "services_classification_v9_new" / "classification_data.json"

# 設定專案根目錄 (讓 Python 能找到 aiva_core 套件)
# 假設腳本位於: .../aiva_core/internal_exploration/python_tools/aiva_cli_implementation.py
# 我們需要將 .../ (aiva_core 的上一層) 加入 sys.path
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ Integration 模組配置：定義基礎路徑（修正：使用正確的專案根目錄）
AIVA_ROOT = CURRENT_DIR.parent.parent.parent.parent  # 向上4層到達 AIVA-git
INTEGRATION_BASE_PATH = AIVA_ROOT / "services" / "integration" / "data" / "internal_exploration"
ANALYSIS_RESULTS_PATH = INTEGRATION_BASE_PATH / "analysis_results" / "external"

try:
    # 正確的 integration 模組路徑
    from services.integration.capability.config import CapabilityConfig
    CLI_OUTPUT_DIR = AIVA_ROOT / "services" / "integration" / "cli_outputs"
    CLI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 使用 Integration 模組 CLI 輸出路徑: {CLI_OUTPUT_DIR}")
except (ImportError, ModuleNotFoundError):
    CLI_OUTPUT_DIR = AIVA_ROOT / "services" / "integration" / "cli_outputs" 
    CLI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 使用預設 Integration CLI 輸出路徑: {CLI_OUTPUT_DIR}")

# 五大模組中文映射（2026-01-03 更新：external_learning 整合至 cognitive_core.learning_system）
MODULE_MAPPING = {
    "cognitive_core": "認知核心模組",
    "internal_exploration": "內探模組", 
    "task_planning": "任務規劃模組",
    "core_capabilities": "核心能力模組",
    "service_backbone": "服務骨幹模組",
    # 向後相容
    "external_learning": "認知核心模組(學習子系統)",
    "learning_system": "認知核心模組(學習子系統)"
}


def find_latest_classification_file() -> Optional[Path]:
    """
    在 integration 目錄中找到最新的分類檔案
    支援多種可能的檔案名稱和位置，自動選擇最新修改的檔案
    
    Returns:
        Path: 最新分類檔案的路徑，如果找不到則返回 None
    """
    # 定義可能的檔案模式和位置
    search_patterns = [
        # 最新的分類檔案（多種可能的名稱）
        INTEGRATION_BASE_PATH / "external_classification.json",
        INTEGRATION_BASE_PATH / "latest_classification.json", 
        INTEGRATION_BASE_PATH / "enriched_classification.json",
        INTEGRATION_BASE_PATH / "*classification*.json",
        
        # analysis_results 目錄下的檔案
        ANALYSIS_RESULTS_PATH / "classification_data.json",
        ANALYSIS_RESULTS_PATH / "latest_classification.json",
        ANALYSIS_RESULTS_PATH / "*classification*.json",
        
        # 可能的其他位置
        INTEGRATION_BASE_PATH / "analysis_history" / "external" / "*classification*.json",
    ]
    
    all_files = []
    for pattern in search_patterns:
        if "*" in str(pattern):
            # 使用 glob 模式
            matched_files = glob(str(pattern))
            all_files.extend([Path(f) for f in matched_files])
        else:
            # 直接檢查檔案存在
            if pattern.exists():
                all_files.append(pattern)
    
    if not all_files:
        return None
    
    # 按修改時間排序，返回最新的檔案
    latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
    return latest_file


class FlowExecutor:
    """
    動態流程執行器
    
    負責讀取 classification_data.json 並執行定義的數據流。
    支援:
    - 動態模組導入與類別實例化
    - 啟發式入口方法偵測
    - Pipeline 數據傳遞
    - Dry Run 模式
    - 文件生成 (Markdown/JSON)
    """
    
    def __init__(self, json_path: Optional[str] = None):
        """
        初始化 FlowExecutor
        
        Args:
            json_path: classification_data.json 的路徑
                      若為 None，優先使用 enriched_classification.json
                      若不存在則使用 latest_classification.json
        """
        # ✅ 自動尋找最新的分類檔案（支援多種可能的輸出檔案）
        latest_file = find_latest_classification_file()
        
        # 備選路徑列表（按優先級排序）
        possible_paths = []
        
        # 如果找到最新檔案，優先使用
        if latest_file:
            possible_paths.append(latest_file)
            
        # 備選的固定路徑
        possible_paths.extend([
            INTEGRATION_BASE_PATH / "external_classification.json",
            ANALYSIS_RESULTS_PATH / "classification_data.json", 
            Path("C:/D/fold7/AIVA-git/features_classification/classification_data.json"),
            DEFAULT_JSON_PATH,
            LATEST_DATA_PATH,
        ])
        
        if json_path:
            self.json_path = json_path
            print(f"[Info] 使用指定數據: {json_path}")
        else:
            for path in possible_paths:
                if path.exists():
                    self.json_path = str(path)
                    # 顯示檔案修改時間以確認使用最新版本
                    mtime = path.stat().st_mtime
                    mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[Info] 使用數據: {path.name} (修改時間: {mod_time})")
                    print(f"[Info] 檔案路徑: {path.parent}")
                    break
            else:
                print("[Error] 找不到任何分類檔案")
                print("嘗試過的路徑:")
                for path in possible_paths:
                    print(f"  - {path}")
                print("\n請確認以下任一檔案存在:")
                print(f"  1. {INTEGRATION_BASE_PATH}/external_classification.json (最新檔案)")
                print(f"  2. {ANALYSIS_RESULTS_PATH}/classification_data.json (分析結果)")
                print(f"  3. 使用 --data 參數指定正確路徑")
                print(f"\n注意: 系統會自動讀取最新修改的檔案")
                self.json_path = str(DEFAULT_JSON_PATH)
        
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, Any]:
        """
        載入分類數據 JSON
        
        Returns:
            包含所有流程定義的字典,若載入失敗則返回空字典
        """
        if not os.path.exists(self.json_path):
            print(f"[Error] 找不到設定檔: {self.json_path}")
            print("請確認檔案是否存在,或使用 --data 參數指定正確路徑。")
            return {}
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] 讀取 JSON 失敗: {e}")
            return {}

    def get_flow_by_id(self, flow_id: int) -> Optional[Dict[str, Any]]:
        """
        根據 ID 獲取 Flow 定義
        
        Args:
            flow_id: 流程 ID
            
        Returns:
            流程定義字典,若不存在則返回 None
        """
        for flow in self.data.get("flows", []):
            if flow["id"] == flow_id:
                return flow
        return None
    
    def search_capabilities(self, query: str, by: str = "all") -> List[Dict[str, Any]]:
        """
        搜尋能力
        
        Args:
            query: 搜尋關鍵字
            by: 搜尋範圍 ("all", "name", "tag", "module", "description")
            
        Returns:
            符合條件的 Flow 列表
        """
        results = []
        query_lower = query.lower()
        
        for flow in self.data.get("flows", []):
            capability = flow.get("capability")
            if not capability:
                continue
            
            matched = False
            
            if by in ["all", "name"]:
                if query_lower in capability.get("name", "").lower():
                    matched = True
            
            if by in ["all", "tag"]:
                tags = capability.get("tags", [])
                if any(query_lower in tag.lower() for tag in tags):
                    matched = True
            
            if by in ["all", "module"]:
                if query_lower in capability.get("module", "").lower():
                    matched = True
                if query_lower in capability.get("module_zh", "").lower():
                    matched = True
            
            if by in ["all", "description"]:
                if query_lower in capability.get("description", "").lower():
                    matched = True
            
            if matched:
                results.append(flow)
        
        return results
    
    def show_capability_info(self, flow_id: int):
        """
        顯示 Flow 的詳細能力資訊
        
        Args:
            flow_id: Flow ID
        """
        flow = self.get_flow_by_id(flow_id)
        if not flow:
            print(f"[Error] Flow {flow_id} 不存在")
            return
        
        capability = flow.get("capability")
        if not capability:
            print(f"[Warning] Flow {flow_id} 無能力資訊（可能使用舊版數據）")
            print(f"路徑: {' -> '.join(flow['path'])}")
            return
        
        print(f"\n{'='*60}")
        print(f"Flow {flow_id}: {capability['name']}")
        print(f"{'='*60}")
        print(f"\n📋 描述:")
        print(f"   {capability['description']}")
        print(f"\n🔧 CLI 指令:")
        print(f"   {capability['command_template']}")
        print(f"\n🏷️  標籤: {', '.join(capability['tags'])}")
        print(f"📦 模組: {capability['module_zh']} ({capability['module']})")
        print(f"📊 複雜度: {capability['complexity']}")
        print(f"🔀 流程長度: {flow['length']} 步")
        print(f"\n📍 完整路徑:")
        for i, step in enumerate(flow['path'], 1):
            print(f"   {i}. {step}")
        print(f"\n💡 執行方式:")
        print(f"   python aiva_cli_implementation.py --flow {flow_id}")
        print(f"{'='*60}\n")

    def _full_path_to_module(self, full_path: str) -> Optional[str]:
        """
        將絕對檔案路徑轉換為 Python 模組路徑
        
        例如: C:\\...\\aiva_core\\external_learning\\xx.py 
              -> aiva_core.external_learning.xx
        
        Args:
            full_path: Windows 絕對檔案路徑
            
        Returns:
            Python 模組路徑,若無法轉換則返回 None
        """
        # 統一將 Windows 反斜線轉換為正斜線,方便處理
        path_str = str(full_path).replace("\\", "/")
        
        # 關鍵: 找到 'aiva_core' 在路徑中的位置作為錨點
        target_marker = "aiva_core"
        if target_marker in path_str:
            # 取得從 aiva_core 開始的子路徑
            idx = path_str.find(target_marker)
            rel_path = path_str[idx:]
            
            # 去掉 .py 副檔名
            if rel_path.endswith(".py"):
                rel_path = rel_path[:-3]
                
            # 將路徑分隔符換成 Python 的模組分隔符 '.'
            return rel_path.replace("/", ".")
            
        return None

    def _snake_to_camel(self, snake_str: str) -> str:
        """
        將 snake_case (檔案名) 轉為 CamelCase (類別名)
        
        例如: scalable_bio_trainer -> ScalableBioTrainer
        
        Args:
            snake_str: snake_case 格式的字串
            
        Returns:
            CamelCase 格式的字串
        """
        return ''.join(x.title() for x in snake_str.split('_'))

    def _find_entry_method(self, instance: Any, step_index: int) -> Optional[Any]:
        """
        啟發式尋找類別的入口方法
        
        根據命名慣例與步驟順序來猜測應該呼叫哪個函數:
        - 常用執行入口: execute, process, run, analyze, start
        - 第一步優先: train, generate, load, initialize
        - 最後一步可能: save, export, output
        
        Args:
            instance: 類別實例
            step_index: 當前步驟索引 (從 0 開始)
            
        Returns:
            找到的方法物件,若找不到則返回 None
        """
        # 獲取該實例所有公開的方法名稱
        methods = [m[0] for m in inspect.getmembers(instance, predicate=inspect.ismethod) 
                   if not m[0].startswith('_')]
        
        # 定義優先順序列表
        # 1. 常用執行入口
        priority_methods = ['execute', 'process', 'run', 'analyze', 'start']
        
        # 2. 如果是第一步,通常涉及初始化或訓練
        if step_index == 0:
            priority_methods = ['train', 'generate', 'load', 'initialize'] + priority_methods
        
        # 3. 如果是最後一步,可能涉及儲存或輸出
        # (這裡暫不實作特定邏輯,通用優先順序通常足夠)

        for method_name in priority_methods:
            if method_name in methods:
                return getattr(instance, method_name)
        
        # 如果常用的都沒找到,嘗試尋找任何看起來像主邏輯的方法 (排除 helper)
        # 這裡可以根據專案具體情況擴充
        return None

    def generate_reference_docs(self, output_format: str = "md", output_dir: Optional[str] = None) -> None:
        """
        生成完整的 CLI 指令參考文件
        
        支援兩種格式:
        - Markdown (.md): 適合人類閱讀,包含表格與格式化
        - JSON (.json): 適合 AI 檢索,結構化數據
        
        ⭐ 輸出位置：默認使用配置的 CLI_OUTPUT_DIR（Integration 模組）
        
        Args:
            output_format: 輸出格式,'md' 或 'json'
            output_dir: 自定義輸出目錄，若為 None 則使用 CLI_OUTPUT_DIR
        """
        flows = self.data.get("flows", [])
        
        # ⭐ 使用配置化的輸出目錄
        if output_dir is None:
            output_dir = str(CLI_OUTPUT_DIR)
        
        # 確保目錄存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if output_format == "md":
            filename = str(output_path / "CLI_COMMANDS_REFERENCE.md")
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("# AIVA Core CLI 指令參考手冊\n\n")
                    f.write(f"> 生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"> 來源設定檔: {os.path.basename(self.json_path)}\n")
                    f.write(f"> 總流程數: {len(flows)}\n\n")
                    
                    f.write("## 快速指令索引\n\n")
                    f.write("此表格列出所有可用流程及其對應的 CLI 執行指令。AI 代理可根據需求檢索此表。\n\n")
                    f.write("| ID | 任務路徑 (Path) | 主要模組 | CLI 指令 |\n")
                    f.write("|:---:|---|---|---|\n")
                    
                    for flow in flows:
                        f_id = flow['id']
                        # 簡化顯示路徑 (只取頭尾或前三個,避免過長)
                        path_str = " -> ".join(flow['path'])
                        
                        # 找出主要模組
                        primary_mod = flow.get('primary_module', 'unknown')
                        
                        # 生成指令
                        cli_command = f"`python -m aiva_core.internal_exploration.aiva_cli_implementation --flow {f_id}`"
                        
                        f.write(f"| {f_id} | {path_str} | {primary_mod} | {cli_command} |\n")
                
                print(f"[Success] Markdown 參考文件已生成: {filename}")
                
            except IOError as e:
                print(f"[Error] 無法寫入文件: {e}")

        elif output_format == "json":
            filename = str(output_path / "cli_commands_db.json")
            db_records = []
            for flow in flows:
                record = {
                    "flow_id": flow['id'],
                    "description": f"Execute flow: {' -> '.join(flow['path'])}",
                    "cli_command": f"python -m aiva_core.internal_exploration.aiva_cli_implementation --flow {flow['id']}",
                    "modules_involved": list(set(flow.get('modules', []))),
                    "steps": flow.get('path', []),
                    "length": flow.get('length', 0)
                }
                db_records.append(record)
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(db_records, f, indent=2, ensure_ascii=False)
                print(f"[Success] JSON 資料庫文件已生成: {filename}")
            except IOError as e:
                print(f"[Error] 無法寫入文件: {e}")

    def execute_flow(self, flow_id: int, context_data: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> None:
        """
        執行指定 ID 的數據流
        
        執行流程:
        1. 讀取流程定義
        2. 對每個步驟:
           a. 解析模組路徑與類別名稱
           b. 動態導入模組
           c. 實例化類別
           d. 尋找並執行入口方法
           e. 在步驟間傳遞輸出數據 (Pipeline)
        3. 顯示最終輸出
        
        Args:
            flow_id: 要執行的流程 ID
            context_data: 初始上下文數據（將在第一步傳入）
            dry_run: 若為 True,僅顯示執行計畫,不實際載入模組或執行
        """
        flow = self.get_flow_by_id(flow_id)
        if not flow:
            print(f"[Error] Flow ID {flow_id} 不存在。請使用 --list 查看可用 ID。")
            return

        # 顯示能力資訊
        capability = flow.get("capability")
        if capability:
            print(f"\n{'='*60}")
            print(f"🚀 準備執行 Flow {flow_id}: {capability['name']}")
            print(f"{'='*60}")
            print(f"📋 能力描述: {capability['description']}")
            print(f"🏷️  標籤: {', '.join(capability['tags'])}")
            print(f"📊 複雜度: {capability['complexity']} | 流程長度: {flow['length']} 步")
            print(f"📍 路徑: {' -> '.join(flow['path'])}")
            print(f"{'='*60}\n")
        else:
            print(f"\n=== [AIVA Flow Executor] 開始執行 Flow {flow_id} ===")
            print(f"路徑預覽: {' -> '.join(flow['path'])}")
            print("-" * 50)
        
        if dry_run:
            print("[Info] Dry Run 模式開啟:僅生成執行計畫,不載入模組或執行代碼。")
        
        # 使用傳入的 context_data，若無則初始化為空字典
        if context_data is None:
            context_data = {}
        elif context_data:
            print(f"[Info] 初始上下文數據: {context_data}")

        for idx, step_info in enumerate(flow.get("classifications", [])):
            script_name = step_info["script"]
            full_path = flow["full_path"][idx]
            
            # 解析模組路徑與類別名稱
            module_path = self._full_path_to_module(full_path)
            class_name = self._snake_to_camel(script_name)

            print(f"\n>> [Step {idx+1}/{len(flow['path'])}] {script_name}")
            print(f"   - File: {full_path}")
            print(f"   - Module: {module_path}")
            print(f"   - Class: {class_name}")

            # Dry Run 模式下,到此為止
            if dry_run:
                continue

            # 實際執行邏輯
            if not module_path:
                print("   [Error] 無法解析模組路徑,跳過此步驟。")
                continue

            try:
                # 1. 動態導入模組
                module = importlib.import_module(module_path)
                
                # 2. 獲取類別
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                else:
                    # 容錯:如果找不到 CamelCase 類別,嘗試搜尋模組內定義的任何類別
                    print(f"   [Warning] 模組內找不到 {class_name},嘗試搜尋其他類別...")
                    classes = [m[1] for m in inspect.getmembers(module, inspect.isclass) 
                               if m[1].__module__ == module.__name__]
                    if classes:
                        cls = classes[0]
                        print(f"   -> 改用類別: {cls.__name__}")
                    else:
                        raise ImportError(f"在 {module_path} 中找不到任何可用的類別")

                # 3. 實例化類別
                print(f"   -> 正在實例化 {cls.__name__}...")
                instance = cls()
                
                # 4. 尋找並執行入口方法
                method = self._find_entry_method(instance, idx)
                
                if method:
                    print(f"   -> 呼叫方法: {method.__name__}()")
                    
                    # 檢查該方法是否接受參數
                    sig = inspect.signature(method)
                    params = sig.parameters
                    
                    try:
                        # 簡單的 Pipeline 邏輯:
                        # 如果上一步有輸出 (context_data),且當前方法接受參數,則傳入
                        if context_data is not None and len(params) > 0:
                            # 假設第一個參數就是輸入數據
                            print("      (傳遞上一步驟的輸出作為參數)")
                            result = method(context_data)
                        else:
                            result = method()
                        
                        # 更新 Context
                        # 如果有回傳值,就更新 context;否則保留上一步的數據
                        if result is not None:
                            context_data = result
                            print(f"   -> 執行成功,獲得輸出: {type(result).__name__}")
                        else:
                            print("   -> 執行成功,無回傳值")
                            
                    except Exception as e:
                        print(f"   [Execution Error] 執行方法時發生錯誤: {e}")
                        # 選擇是否要中斷流程
                        break
                else:
                    print("   [Warning] 找不到適合的入口方法 (如 execute, train, run),跳過執行。")
                    
            except ImportError as e:
                print(f"   [Import Error] 導入模組失敗: {e}")
                break
            except Exception as e:
                print(f"   [System Error] 發生未預期的錯誤: {e}")
                break

        if not dry_run:
            print(f"\n=== Flow {flow_id} 執行完畢 ===")
            if context_data:
                print(f"最終輸出: {context_data}")


def main():
    """
    主程式入口 - CLI 參數解析與功能路由
    """
    parser = argparse.ArgumentParser(
        description="AIVA Core 動態流程執行器與文件生成工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--flow", 
        type=int, 
        help="執行指定 ID 的數據流。\n範例: python aiva_cli_implementation.py --flow 11"
    )
    
    parser.add_argument(
        "--generate-doc", 
        choices=['md', 'json'], 
        help="生成指令參考文件。\n"
             "  md   : 生成 Markdown 文件 (適合人類閱讀)\n"
             "  json : 生成 JSON 資料庫 (適合 AI 檢索)"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="列出前 20 個可用的 Flow ID 與路徑"
    )
    
    parser.add_argument(
        "--menu",
        action="store_true",
        help="啟動互動式選單 (按模組和能力瀏覽)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="[配合 --flow 使用] 僅顯示執行計畫,不實際載入模組或運行代碼"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"指定 classification_data.json 的路徑\n預設: 自動讀取最新的分類檔案 (integration 目錄中最新修改的 *classification*.json)\n支援多種檔案名稱: external_classification.json, latest_classification.json, enriched_classification.json 等"
    )

    args = parser.parse_args()
    
    # 初始化執行器
    executor = FlowExecutor(json_path=args.data)
    
    # 如果沒有任何參數，預設啟動互動選單
    if not any([args.flow, args.generate_doc, args.list, args.menu]):
        InteractiveMenu(executor).run()
        return
    
    if args.menu:
        InteractiveMenu(executor).run()
            
    elif args.list:
        print(f"載入設定檔: {executor.json_path}")
        print("可用 Flow 列表 (顯示前 20 筆):")
        print("-" * 60)
        flows = executor.data.get("flows", [])
        for f in flows[:20]:
            print(f"ID: {f['id']:<4} | Len: {f.get('length', 0)} | Path: {' -> '.join(f['path'])}")
        if len(flows) > 20:
            print(f"... (還有 {len(flows) - 20} 條流程)")
            
    elif args.generate_doc:
        print(f"正在生成 {args.generate_doc.upper()} 格式的參考文件...")
        executor.generate_reference_docs(output_format=args.generate_doc)
        
    elif args.flow:
        executor.execute_flow(args.flow, dry_run=args.dry_run)


class InteractiveMenu:
    """
    互動式選單介面
    
    提供雙層聚合顯示:
    1. 第一層: 按六大模組分類
    2. 第二層: 按「起點->終點」聚合能力
    3. 第三層: 顯示具體路徑變體
    """
    
    def __init__(self, executor: FlowExecutor):
        self.executor = executor
        self.flows = executor.data.get("flows", [])
        self._build_indexes()

    def _build_indexes(self):
        """建立雙層索引: [Module][(Start, End)] -> [Flows]"""
        from collections import defaultdict
        self.module_groups = defaultdict(lambda: defaultdict(list))
        
        for flow in self.flows:
            mod = flow.get("primary_module", "other")
            start = flow.get("start", "?")
            end = flow.get("end", "?")
            self.module_groups[mod][(start, end)].append(flow)

    def clear(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def run(self):
        """主選單循環"""
        while True:
            self.clear()
            meta = self.executor.data.get("metadata", {})
            total = meta.get("total_flows", len(self.flows))
            
            print("=" * 60)
            print(f" AIVA CORE - 任務控制台 (Flows: {total})")
            print("=" * 60)
            
            # 按六大模組順序顯示
            all_mods = sorted(
                self.module_groups.keys(), 
                key=lambda x: list(MODULE_MAPPING.keys()).index(x) if x in MODULE_MAPPING else 99
            )

            for i, mod in enumerate(all_mods):
                cn_name = MODULE_MAPPING.get(mod, mod)
                unique_caps = len(self.module_groups[mod])
                total_flows = sum(len(v) for v in self.module_groups[mod].values())
                
                print(f" [{i+1}] {cn_name:<15} | {unique_caps} 種能力 ({total_flows} 路徑)")
            
            print(" [0] 離開")
            choice = input("\n> 請選擇模組: ").strip()
            
            if choice == "0":
                break
            if choice.isdigit() and 0 < int(choice) <= len(all_mods):
                self._show_capability_menu(all_mods[int(choice)-1])

    def _show_capability_menu(self, module_name: str):
        """顯示能力的聚合列表 (起點->終點)"""
        caps = self.module_groups[module_name]
        sorted_caps = sorted(caps.keys())
        
        while True:
            self.clear()
            cn_name = MODULE_MAPPING.get(module_name, module_name)
            print(f"\n=== {cn_name} 能力清單 ===")
            print(f"{'No.':<4} | {'起點 (Start)':<25} -> {'終點 (End)':<25} | {'變體數'}")
            print("-" * 80)
            
            for i, (start, end) in enumerate(sorted_caps):
                count = len(caps[(start, end)])
                print(f" [{i+1:<2}] | {start:<25} -> {end:<25} | {count}")
            
            print("-" * 80)
            print(" [b] 返回  [輸入數字] 選擇能力")
            
            choice = input("\n> ").strip()
            if choice == 'b':
                break
            
            if choice.isdigit() and 0 < int(choice) <= len(sorted_caps):
                key = sorted_caps[int(choice)-1]
                self._show_variant_menu(key, caps[key])

    def _show_variant_menu(self, cap_key: tuple, flows: list):
        """顯示具體路徑變體"""
        while True:
            self.clear()
            print(f"\n=== 執行路徑: {cap_key[0]} -> {cap_key[1]} ===")
            print("-" * 80)
            
            # 按長度排序
            flows.sort(key=lambda x: x['length'])
            
            for i, flow in enumerate(flows):
                mid = flow['path'][1:-1]
                mid_str = " -> ".join(mid) if mid else "(直連)"
                print(f" [{i+1}] Flow {flow['id']} (步數:{flow['length']}): ... -> {mid_str} -> ...")
            
            print("-" * 80)
            print(" [b] 返回  [輸入數字] 執行  [v 數字] 預覽")
            
            cmd = input("\n> ").strip()
            if cmd == 'b':
                break
            
            is_view = False
            idx = -1
            
            if cmd.startswith('v '):
                is_view = True
                try:
                    idx = int(cmd.split()[1]) - 1
                except (ValueError, IndexError):
                    pass
            elif cmd.isdigit():
                idx = int(cmd) - 1
                
            if 0 <= idx < len(flows):
                target = flows[idx]
                print("\n" + "=" * 30)
                self.executor.execute_flow(target['id'], dry_run=is_view)
                input("\n[按 Enter 繼續...]")


if __name__ == "__main__":
    main()
