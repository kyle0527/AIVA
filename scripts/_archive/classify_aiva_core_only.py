#!/usr/bin/env python3
"""
aiva_core 專用能力分類器

只處理 core/aiva_core 模組的能力數據，避免循環導入問題
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# 添加路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "services"))

print("=" * 60)
print("🎯 aiva_core 專用能力分類器")
print("=" * 60)

# 手動實現分類邏輯（避免導入 internal_loop_connector）
class SimpleCapabilityScopeClassifier:
    """簡化版分類器 - 避免循環導入"""
    
    def classify_scope(self, file_path: str) -> tuple[str, str]:
        """分類範圍和可見性"""
        file_path_lower = file_path.lower()
        
        # CORE + INTERNAL: 內部探索、內部閉環
        if ("internal_exploration" in file_path_lower or 
            "internal_loop" in file_path_lower):
            return ("CORE", "INTERNAL")
        
        # CORE + INTERNAL: 認知核心內部組件
        if "cognitive_core" in file_path_lower:
            if any(x in file_path_lower for x in ["neural", "decision", "rag", "memory"]):
                return ("CORE", "INTERNAL")
            return ("CORE", "PUBLIC")
        
        # SERVICE + PUBLIC: 對外服務能力
        if any(x in file_path_lower for x in ["task_planning", "core_capabilities"]):
            return ("SERVICE", "PUBLIC")
        
        # SERVICE + SYSTEM: 服務骨架
        if "service_backbone" in file_path_lower:
            return ("SERVICE", "SYSTEM")
        
        # 默認
        return ("CORE", "PUBLIC")
    
    def classify_access_level(self, category: str) -> str:
        """分類訪問級別"""
        category_lower = category.lower()
        
        if category_lower in ["scanning", "attacking"]:
            return "L2_MODULE"
        if category_lower in ["analysis", "reporting"]:
            return "L1_SERVICE"
        if category_lower == "integration":
            return "L0_SYSTEM"
        return "L3_INTERNAL"
    
    def detect_available_in(self, _file_path: str) -> list[str]:
        """檢測可用路徑
        
        Args:
            _file_path: 檔案路徑（保留用於未來擴展）
        """
        return ["core"]  # aiva_core 只在 core 可用
    
    def detect_cli_info(self, _cap: dict) -> tuple[bool, str | None, str]:
        """檢測 CLI 信息
        
        Args:
            _cap: 能力字典（保留用於未來擴展）
        """
        # aiva_core 內部能力不是 CLI 命令
        # CLI 命令應該在 features 或 scan 模組
        return (False, None, "NONE")

# 載入數據
input_dir = project_root / "data" / "analysis_results"
input_file = sorted(input_dir.glob("capabilities_aiva_core_only_*.json"))[-1]

print(f"\n📂 載入數據: {input_file.name}")
with open(input_file, 'r', encoding='utf-8') as f:
    capabilities = json.load(f)

print(f"   ✅ 載入 {len(capabilities)} 個能力")

# 初始化分類器
print("\n🔧 初始化簡化版分類器...")
classifier = SimpleCapabilityScopeClassifier()

# 分類
print("\n🔄 開始分類...")
classified = []
for i, cap in enumerate(capabilities, 1):
    scope, visibility = classifier.classify_scope(cap['file_path'])
    access_level = classifier.classify_access_level(cap.get('category', ''))
    available_in = classifier.detect_available_in(cap['file_path'])
    has_cli, cli_command, cli_maturity = classifier.detect_cli_info(cap)
    
    cap['scope'] = scope
    cap['visibility'] = visibility
    cap['access_level'] = access_level
    cap['available_in'] = available_in
    cap['has_cli'] = has_cli
    cap['cli_command'] = cli_command
    cap['cli_maturity'] = cli_maturity
    cap['service_dependencies'] = []
    
    classified.append(cap)
    
    if i % 50 == 0:
        print(f"   處理中: {i}/{len(capabilities)}")

# 保存
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = input_dir / f"capabilities_aiva_core_classified_{timestamp}.json"

print("\n💾 保存分類結果...")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(classified, f, indent=2, ensure_ascii=False)

print(f"   ✅ 保存完成: {output_path.name}")

# 生成報告
print("\n📊 生成統計報告...")
scopes = Counter(c['scope'] for c in classified)
visibilities = Counter(c['visibility'] for c in classified)
access_levels = Counter(c['access_level'] for c in classified)
cli_count = sum(1 for c in classified if c['has_cli'])

report_path = input_dir / f"aiva_core_classification_report_{timestamp}.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# aiva_core 能力分類報告\n\n")
    f.write(f"**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**總能力數**: {len(classified)}\n\n")
    
    f.write("## 範圍分佈 (Scope)\n\n")
    for scope, count in scopes.most_common():
        percentage = count / len(classified) * 100
        f.write(f"- **{scope}**: {count} ({percentage:.1f}%)\n")
    
    f.write("\n## 可見性分佈 (Visibility)\n\n")
    for vis, count in visibilities.most_common():
        percentage = count / len(classified) * 100
        f.write(f"- **{vis}**: {count} ({percentage:.1f}%)\n")
    
    f.write("\n## 訪問級別分佈 (Access Level)\n\n")
    for level, count in access_levels.most_common():
        percentage = count / len(classified) * 100
        f.write(f"- **{level}**: {count} ({percentage:.1f}%)\n")
    
    f.write("\n## CLI 能力\n\n")
    f.write(f"- **CLI 可調用**: {cli_count}\n")
    f.write(f"- **僅內部使用**: {len(classified) - cli_count}\n")

print(f"   ✅ 報告生成完成: {report_path.name}")

# 控制台輸出
print("\n" + "=" * 60)
print("✅ 分類完成！")
print("=" * 60)

print("\n📊 統計總結:")
print(f"   總能力數量: {len(classified)}")
print("   處理時間: 完成")

print("\n範圍分佈:")
for scope, count in scopes.most_common():
    percentage = count / len(classified) * 100
    print(f"   {scope}: {count} ({percentage:.1f}%)")

print("\n可見性分佈:")
for vis, count in visibilities.most_common():
    percentage = count / len(classified) * 100
    print(f"   {vis}: {count} ({percentage:.1f}%)")

print(f"\nCLI 能力: {cli_count}")

print("\n📁 輸出文件:")
print(f"   分類結果: {output_path.name}")
print(f"   報告: {report_path.name}")
