"""
AIVA Exploration Pipeline - 認知更新管線 (v3.1)

這是 AIVA 系統自我認知更新的總控腳本。

核心功能:
1. 代碼結構分析 (Analyzer)
2. 數據流分類 (Classifier)  
3. 版本差異比對 (Diff)
4. 自動版本控制 (Versioning)

使用方式:
    # 分析特定模組
    python aiva_exploration_pipeline.py --target cognitive_core
    
    # 分析整個 core
    python aiva_exploration_pipeline.py --target core
    
    # 分析所有模組
    python aiva_exploration_pipeline.py --target all

輸出結構:
    analysis_history/
    ├── v1/
    │   ├── analysis_results.json
    │   ├── classification_data.json
    │   └── diff_report.md
    ├── v2/
    └── ...
    
    latest_classification.json  # 始終指向最新版本
"""

import sys
import os
import json
import logging
import shutil
import glob
import argparse
from pathlib import Path
from datetime import datetime

# ==========================================
# 設定日誌與路徑
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [Pipeline] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("AIVA_Pipeline")

# 目錄結構說明:
# parent(1):   internal_exploration/python_tools/
# parent(2):   internal_exploration/
# parent(3):   aiva_core/
# parent(4):   core/
# parent(5):   services/
# parent(6):   專案根目錄
CURRENT_DIR = Path(__file__).resolve().parent
SERVICES_ROOT = CURRENT_DIR.parent.parent.parent.parent  # 到達 services/
PROJECT_ROOT = SERVICES_ROOT.parent  # 到達專案根目錄

if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# 統一資料存放路徑配置（依據 README.md 文檔）
# 所有分析結果統一存放在: services/integration/data/internal_exploration/
# ==========================================
INTEGRATION_DATA_DIR = SERVICES_ROOT / "integration" / "data" / "internal_exploration"
ANALYSIS_HISTORY_DIR = INTEGRATION_DATA_DIR / "analysis_history"
LATEST_CLASSIFICATION_JSON = INTEGRATION_DATA_DIR / "latest_classification.json"
ANALYSIS_RESULTS_DIR = INTEGRATION_DATA_DIR / "analysis_results"

HISTORY_DIR = ANALYSIS_HISTORY_DIR
LATEST_DATA_LINK = LATEST_CLASSIFICATION_JSON
logger.info(f"✅ 資料路徑: {HISTORY_DIR}")

# 導入功能模組 - 使用直接文件導入避免任何循環依賴
try:
    import importlib.util
    
    # 載入 analyzer（已升級為 v2 代碼）
    analyzer_path = CURRENT_DIR / "aiva_flow_analyzer.py"
    if not analyzer_path.exists():
        raise FileNotFoundError(f"找不到 analyzer: {analyzer_path}")
    
    spec_analyzer = importlib.util.spec_from_file_location("aiva_flow_analyzer_module", analyzer_path)
    if spec_analyzer is None or spec_analyzer.loader is None:
        raise ImportError(f"無法創建 analyzer spec: {analyzer_path}")
    
    analyzer_module = importlib.util.module_from_spec(spec_analyzer)
    sys.modules['aiva_flow_analyzer_module'] = analyzer_module
    spec_analyzer.loader.exec_module(analyzer_module)
    AIVAFlowAnalyzer = analyzer_module.AIVAFlowAnalyzer
    
    # 載入內部分類器 (aiva_core)
    internal_classifier_path = CURRENT_DIR / "aiva_flow_classifier.py"
    if not internal_classifier_path.exists():
        raise FileNotFoundError(f"找不到內部分類器: {internal_classifier_path}")
    
    spec_internal = importlib.util.spec_from_file_location("aiva_flow_classifier_module", internal_classifier_path)
    if spec_internal is None or spec_internal.loader is None:
        raise ImportError(f"無法創建內部分類器 spec: {internal_classifier_path}")
    
    internal_module = importlib.util.module_from_spec(spec_internal)
    sys.modules['aiva_flow_classifier_module'] = internal_module
    spec_internal.loader.exec_module(internal_module)
    AIVAFlowClassifier = internal_module.AIVAFlowClassifier
    
    # 載入外部分類器 (features/scan)
    external_classifier_path = CURRENT_DIR / "aiva_external_module_classifier.py"
    if not external_classifier_path.exists():
        raise FileNotFoundError(f"找不到外部分類器: {external_classifier_path}")
    
    spec_external = importlib.util.spec_from_file_location("aiva_external_classifier_module", external_classifier_path)
    if spec_external is None or spec_external.loader is None:
        raise ImportError(f"無法創建外部分類器 spec: {external_classifier_path}")
    
    external_module = importlib.util.module_from_spec(spec_external)
    sys.modules['aiva_external_classifier_module'] = external_module
    spec_external.loader.exec_module(external_module)
    ExternalModuleClassifier = external_module.ExternalModuleClassifier
    
    logger.info("✅ 模組載入成功（內部+外部分類器）")
    
except (ImportError, FileNotFoundError, AttributeError) as e:
    logger.error(f"❌ 模組導入失敗: {e}")
    logger.error(f"   當前目錄: {CURRENT_DIR}")
    logger.error(f"   Analyzer 路徑: {analyzer_path if 'analyzer_path' in locals() else 'N/A'}")
    # classifier_path 變量不存在，使用 internal_classifier_path 和 external_classifier_path
    logger.error(f"   Internal Classifier 路徑: {internal_classifier_path if 'internal_classifier_path' in locals() else 'N/A'}")
    logger.error(f"   External Classifier 路徑: {external_classifier_path if 'external_classifier_path' in locals() else 'N/A'}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class ExplorationPipeline:
    """AIVA 探索管線控制器"""
    
    def __init__(self, target_path, target_module="core", depth=10):
        """
        初始化管線
        
        Args:
            target_path (str): 要分析的目標路徑
                - 'all': 分析所有模組
                - 'core': 分析 services/core
                - 'cognitive_core': 分析 core/aiva_core/cognitive_core
                - 或任何相對/絕對路徑
            target_module (str): 目標模組名稱 (用於分類儲存)
                - 'core': 核心模組
                - 'features': 功能模組
                - 'scan': 掃描模組
                - 'integration': 整合模組
            depth (int): AST 分析的遞迴深度 (預設: 10)
        """
        self.target_path = target_path
        self.target_module = target_module
        self.depth = depth
        self.current_version_dir = None
        self.prev_version_dir = None
        
        # 確保歷史目錄存在
        os.makedirs(HISTORY_DIR, exist_ok=True)

    def _get_next_version_dir(self):
        """計算並建立下一個版本目錄 (例如 v5 -> v6)"""
        existing_dirs = glob.glob(str(HISTORY_DIR / "v*"))
        max_ver = 0
        self.prev_version_dir = None

        for d in existing_dirs:
            try:
                name = os.path.basename(d)
                if name.startswith('v') and name[1:].isdigit():
                    ver = int(name[1:])
                    if ver > max_ver:
                        max_ver = ver
                        self.prev_version_dir = Path(d)
            except ValueError:
                continue
        
        next_ver = max_ver + 1
        new_dir = HISTORY_DIR / f"v{next_ver}"
        os.makedirs(new_dir, exist_ok=True)
        return new_dir

    def _resolve_target_path(self):
        """解析目標路徑，返回實際的文件系統路徑"""
        target = self.target_path.lower()
        
        logger.info(f"🔍 解析目標: {self.target_path}")
        logger.info(f"   SERVICES_ROOT: {SERVICES_ROOT}")
        logger.info(f"   目標模組類型: {self.target_module}")
        
        if target == 'all':
            result = SERVICES_ROOT
            logger.info(f"   → 全部模組: {result}")
            return result
        elif target == 'core':
            # ✅ 修正：分析 aiva_core 而不是整個 core 目錄
            result = SERVICES_ROOT / 'core' / 'aiva_core'
            logger.info(f"   → AIVA Core 模組: {result}")
            return result
        elif target == 'features':
            result = SERVICES_ROOT / 'features'
            logger.info(f"   → Features 模組: {result}")
            return result
        elif target == 'scan':
            result = SERVICES_ROOT / 'scan'
            logger.info(f"   → Scan 模組: {result}")
            return result
        elif target.startswith('features_'):
            # 已廢棄：原本的 features_ready 等子目錄已移除，現在所有模組直接位於 features/ 下
            result = SERVICES_ROOT / 'features' / self.target_path
            logger.info(f"   → Features 子模組: {result}")
            return result
        else:
            # 按照優先順序嘗試多個可能的路徑
            candidates = [
                # 1. 作為 aiva_core 下的模組名稱 (內部模組)
                SERVICES_ROOT / 'core' / 'aiva_core' / self.target_path,
                # 2. 作為 features 下的模組 (外部功能模組)
                SERVICES_ROOT / 'features' / self.target_path,
                # 3. 作為 scan 下的模組 (外部掃描模組)
                SERVICES_ROOT / 'scan' / self.target_path,
                # 4. 作為 core 下的路徑
                SERVICES_ROOT / 'core' / self.target_path,
                # 5. 作為 services 下的相對路徑
                SERVICES_ROOT / self.target_path,
                # 6. 作為絕對路徑或當前目錄相對路徑
                Path(self.target_path).resolve(),
            ]
            
            for i, candidate in enumerate(candidates, 1):
                logger.info(f"   嘗試 #{i}: {candidate}")
                if candidate.exists():
                    logger.info(f"   ✅ 找到: {candidate}")
                    return candidate
            
            # 所有嘗試都失敗，返回第一個候選（讓後續邏輯處理錯誤）
            logger.warning(f"   ⚠️ 所有路徑都不存在，使用: {candidates[0]}")
            return candidates[0]

    def run(self):
        """執行完整管線"""
        # 1. 準備環境
        self.current_version_dir = self._get_next_version_dir()
        version_name = self.current_version_dir.name
        resolved_path = self._resolve_target_path()
        
        logger.info("=" * 60)
        logger.info(f"🚀 啟動 AIVA 認知更新管線 | 版本: {version_name}")
        logger.info(f"📂 分析目標: {self.target_path}")
        logger.info(f"📍 解析路徑: {resolved_path}")
        logger.info(f"💾 輸出目錄: {self.current_version_dir}")
        logger.info("=" * 60)

        if not resolved_path.exists():
            logger.error(f"❌ 目標路徑不存在: {resolved_path}")
            return False

        # 定義檔案路徑
        analysis_json = self.current_version_dir / "analysis_results.json"
        classification_json = self.current_version_dir / "classification_data.json"
        diff_report = self.current_version_dir / "diff_report.md"

        # 2. 執行分析 (Analyzer)
        if not self._step_analyze(resolved_path, analysis_json):
            return False

        # 3. 執行分類 (Classifier)
        if not self._step_classify(analysis_json, classification_json):
            return False

        # 4. 執行差異比對 (Diff)
        self._step_generate_diff(classification_json, diff_report)

        # 5. 更新系統指針 (Latest Link)
        self._step_update_link(classification_json)

        # 6. 生成 CLI 指令文檔
        self._step_generate_cli_docs(classification_json)

        logger.info(f"✨ 管線執行完畢。數據已更新至 {version_name}。")
        return True

    def _step_analyze(self, target_path, output_json):
        """步驟 1: 代碼結構分析"""
        logger.info(">> [1/5] 執行代碼結構分析 (Analyzer)...")
        logger.info(f"   目標: {target_path}")
        logger.info(f"   模組: {self.target_module}")
        logger.info(f"   深度: {self.depth}")
        
        try:
            analyzer = AIVAFlowAnalyzer(target_dir=str(PROJECT_ROOT))
            
            # 分析目標路徑 (執行分析但不需要使用返回結果)
            analyzer.analyze_directory(
                target=str(target_path),
                depth=self.depth,
                verbose=False
            )
            
            # 保存結果到版本目錄
            analyzer.save_results(output_dir=str(self.current_version_dir))
            
            if output_json.exists():
                with open(output_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    flow_count = len(data.get('flows', []))
                    logger.info(f"   ✅ 發現 {flow_count} 個數據流")
                return True
            else:
                logger.error("   ❌ 未生成分析結果文件")
                return False
                
        except Exception as e:
            logger.error(f"❌ 分析階段失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _step_classify(self, input_json, output_json):
        """步驟 2: 數據流分類"""
        logger.info(">> [2/5] 執行數據流分類 (Classifier)...")
        
        if not input_json.exists():
            logger.error(f"   ❌ 找不到分析結果: {input_json}")
            return False
        
        try:
            # 根據 target_module 選擇分類器
            if self.target_module == 'core':
                # 使用內部分類器 (aiva_core 6模組)
                logger.info("   使用內部分類器 (AIVAFlowClassifier)")
                classifier = AIVAFlowClassifier(
                    input_dir=str(self.current_version_dir),
                    output_dir=str(self.current_version_dir),
                    verbose=False,
                    merge_duplicate_flows=False  # 不合併：不同路徑 = 不同執行邏輯
                )
            else:
                # 使用外部分類器 (features/scan 功能模組)
                logger.info("   使用外部分類器 (ExternalModuleClassifier)")
                classifier = ExternalModuleClassifier(
                    input_dir=str(self.current_version_dir),
                    output_dir=str(self.current_version_dir),
                    verbose=False
                )
            
            classifier.load_flow_data()
            classifier.classify_flows()
            
            # 只有內部分類器有 analyze_multi_path_endpoints
            if self.target_module == 'core' and hasattr(classifier, 'analyze_multi_path_endpoints'):
                classifier.analyze_multi_path_endpoints()
            
            classifier.generate_reports()
            
            if output_json.exists():
                with open(output_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total = data.get('metadata', {}).get('total_flows', 0)
                    logger.info(f"   ✅ 分類完成: {total} 個流程")
                return True
            else:
                logger.error("   ❌ 未生成分類結果文件")
                return False
                
        except Exception as e:
            logger.error(f"❌ 分類階段失敗: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _step_generate_diff(self, new_file, report_file):
        """步驟 3: 生成版本差異報告"""
        logger.info(">> [3/5] 生成版本差異報告 (Diff)...")
        
        if not self.prev_version_dir:
            logger.info("   (無舊版本，跳過比對)")
            return

        old_file = self.prev_version_dir / "classification_data.json"
        if not old_file.exists():
            logger.info("   (舊版本無分類數據，跳過比對)")
            return

        try:
            with open(old_file, 'r', encoding='utf-8') as f:
                old = json.load(f)
            with open(new_file, 'r', encoding='utf-8') as f:
                new = json.load(f)

            # 使用路徑簽名作為唯一標識（跳過已合併的流程）
            def get_path_sig(flow):
                if 'merged_into' in flow:
                    return None
                path = flow.get('path', flow.get('original_path', []))
                return "->".join(path) if path else None
            
            old_flows = {sig: f for f in old.get('flows', []) if (sig := get_path_sig(f))}
            new_flows = {sig: f for f in new.get('flows', []) if (sig := get_path_sig(f))}

            added = set(new_flows.keys()) - set(old_flows.keys())
            removed = set(old_flows.keys()) - set(new_flows.keys())
            
            # 確保 prev_version_dir 和 current_version_dir 不為 None
            prev_name = self.prev_version_dir.name if self.prev_version_dir else "N/A"
            curr_name = self.current_version_dir.name if self.current_version_dir else "N/A"
            
            # 寫入報告
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# AIVA 能力差異報告\n\n")
                f.write(f"**版本變更**: {prev_name} → {curr_name}\n")
                f.write(f"**分析時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**分析目標**: {self.target_path}\n\n")
                
                f.write("## 📊 變更摘要\n\n")
                f.write(f"- 🟢 新增能力: {len(added)}\n")
                f.write(f"- 🔴 移除/變更: {len(removed)}\n")
                f.write(f"- 📦 總流程數: {len(new_flows)}\n\n")
                
                if added:
                    f.write("## 🟢 新增能力 (New Capabilities)\n\n")
                    for sig in sorted(added):
                        flow = new_flows[sig]
                        # 跳過已合併的流程
                        if 'merged_into' in flow:
                            continue
                        path = flow.get('path', flow.get('original_path', []))
                        f.write(f"### Flow {flow['id']}\n\n")
                        f.write(f"- **模組**: {flow.get('primary_module', 'unknown')}\n")
                        f.write(f"- **長度**: {flow.get('length', len(path) if path else 0)} 步\n")
                        f.write(f"- **路徑**: `{' → '.join(path) if path else 'Unknown'}`\n\n")
                
                if removed:
                    f.write("## 🔴 移除能力 (Removed Capabilities)\n\n")
                    for sig in sorted(removed):
                        flow = old_flows[sig]
                        # 跳過已合併的流程
                        if 'merged_into' in flow:
                            continue
                        path = flow.get('path', flow.get('original_path', []))
                        f.write(f"- **Flow {flow['id']}**: `{' → '.join(path) if path else 'Unknown'}`\n")

            logger.info(f"   ✅ 差異分析完成: +{len(added)} / -{len(removed)}")
            logger.info(f"   📄 報告位置: {report_file}")

        except Exception as e:
            logger.error(f"   ⚠️ 差異比對錯誤: {e}")

    def _step_update_link(self, final_json_path):
        """步驟 4: 更新最新數據鏈接"""
        logger.info(">> [4/5] 更新系統數據指針...")
        
        try:
            if final_json_path.exists():
                # 刪除舊鏈接
                if LATEST_DATA_LINK.exists():
                    LATEST_DATA_LINK.unlink()
                
                # 創建新鏈接（複製而非符號鏈接，避免權限問題）
                shutil.copy2(final_json_path, LATEST_DATA_LINK)
                logger.info(f"   ✅ 已更新 {LATEST_DATA_LINK.name}")
            else:
                logger.error("   ❌ 找不到分類數據文件")
        except Exception as e:
            logger.error(f"   ❌ 更新失敗: {e}")

    # ⚠️ 已移除：_save_to_analysis_data (2026-01-04)
    # 原因：移除 analysis_data/ 中間層，統一使用 data/internal_exploration/
    # 所有結果已直接保存到 analysis_history/v{n}/ 和 latest_classification.json

    def _step_generate_cli_docs(self, classification_json):
        """步驟 5: 生成 CLI 指令文檔"""
        logger.info(">> [5/5] 生成 CLI 指令文檔...")
        
        try:
            # 載入 CLI 實現模組
            cli_impl_path = CURRENT_DIR / "aiva_cli_implementation.py"
            if not cli_impl_path.exists():
                logger.warning(f"   ⚠️ 找不到 CLI 實現模組: {cli_impl_path}")
                return
            
            spec = importlib.util.spec_from_file_location("aiva_cli_impl", cli_impl_path)
            if spec is None or spec.loader is None:
                logger.error("   ❌ 無法創建 CLI 模組 spec")
                return
            
            cli_module = importlib.util.module_from_spec(spec)
            sys.modules['aiva_cli_impl'] = cli_module
            spec.loader.exec_module(cli_module)
            
            # 執行文檔生成
            output_dir = self.current_version_dir
            
            # 使用 FlowExecutor 類來生成文檔
            executor = cli_module.FlowExecutor(str(classification_json))
            
            # 生成 Markdown 文檔
            executor.generate_reference_docs(output_format='md', output_dir=str(output_dir))
            logger.info("   ✅ Markdown 文檔: CLI_COMMANDS_REFERENCE.md")
            
            # 生成 JSON 資料庫
            executor.generate_reference_docs(output_format='json', output_dir=str(output_dir))
            logger.info("   ✅ JSON 資料庫: cli_commands_db.json")
            
        except Exception as e:
            logger.error(f"   ⚠️ CLI 文檔生成失敗: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="AIVA 認知更新管線 - 自動化代碼分析與版本控制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 分析 core 模組 (使用內部分類器)
  python aiva_exploration_pipeline.py --target core --module core
  
  # 分析 cognitive_core 子模組 (使用內部分類器)
  python aiva_exploration_pipeline.py --target cognitive_core --module core
  
  # 分析 features 目錄 (使用外部分類器) - 注意：features_ready 等子目錄已移除
  python aiva_exploration_pipeline.py --target features --module features
  
  # 分析整個 features 目錄 (使用外部分類器)
  python aiva_exploration_pipeline.py --target features --module features
  
  # 分析 scan 模組 (使用外部分類器)
  python aiva_exploration_pipeline.py --target scan --module scan
        """
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="aiva_core",
        help="指定要分析的目標路徑 (預設: aiva_core)"
    )
    
    parser.add_argument(
        "--module", "-m",
        type=str,
        default="core",
        choices=["core", "features", "scan", "integration"],
        help="指定目標模組名稱，用於分類儲存 (預設: core)"
    )
    
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=10,
        help="AST 分析的遞迴深度 (預設: 10)"
    )
    
    args = parser.parse_args()
    
    pipeline = ExplorationPipeline(
        target_path=args.target,
        target_module=args.module,
        depth=args.depth
    )
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
