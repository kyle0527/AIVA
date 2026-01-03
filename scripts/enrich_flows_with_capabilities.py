#!/usr/bin/env python3
"""
Flow 能力資訊擴充腳本
為 latest_classification.json 的 840 個 flows 自動添加能力描述和 CLI 資訊

目標：讓 AI 和人類都能理解每個 Flow 的能力
- 從檔案路徑推斷功能
- 生成 CLI 指令模板
- 添加人類可讀描述
- 保持向後兼容（不破壞現有執行器）

輸出：enriched_classification.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class CapabilityEnricher:
    """Flow 能力資訊擴充器"""
    
    def __init__(self, input_json: str, output_json: str = None):
        self.input_path = Path(input_json)
        self.output_path = Path(output_json) if output_json else self.input_path.parent / "enriched_classification.json"
        
        # 模組中文映射
        self.module_names = {
            "cognitive_core": "認知核心",
            "internal_exploration": "內探模組",
            "task_planning": "任務規劃",
            "external_learning": "外學模組",
            "core_capabilities": "核心能力",
            "service_backbone": "服務骨幹"
        }
        
        # 常見功能關鍵字映射
        self.capability_patterns = {
            # 掃描類
            r"scan|scanner": ("掃描", "scan", ["recon", "discovery"]),
            r"xss": ("XSS漏洞檢測", "xss_detection", ["web", "vulnerability", "attack"]),
            r"sqli|sql.*injection": ("SQL注入檢測", "sqli_detection", ["web", "vulnerability", "database"]),
            r"port": ("端口掃描", "port_scan", ["network", "recon"]),
            r"exploit": ("漏洞利用", "exploitation", ["attack", "offensive"]),
            
            # 分析類
            r"analyzer?|analysis": ("資料分析", "analysis", ["data", "intelligence"]),
            r"parser?": ("資料解析", "parsing", ["data", "processing"]),
            r"classifier": ("分類器", "classification", ["ml", "ai"]),
            r"detector?|detect": ("檢測器", "detection", ["security", "monitoring"]),
            
            # 執行類
            r"executor?|execute": ("執行器", "execution", ["runtime", "orchestration"]),
            r"worker": ("工作器", "worker", ["processing", "task"]),
            r"handler": ("處理器", "handler", ["processing", "event"]),
            r"runner": ("運行器", "runner", ["execution"]),
            
            # 管理類
            r"manager": ("管理器", "manager", ["state", "lifecycle"]),
            r"coordinator": ("協調器", "coordinator", ["orchestration", "control"]),
            r"orchestrator": ("編排器", "orchestrator", ["workflow", "automation"]),
            r"scheduler": ("調度器", "scheduler", ["timing", "queue"]),
            
            # 狀態類
            r"session.*state": ("會話狀態", "session_state", ["state", "context"]),
            r"state": ("狀態管理", "state_management", ["state", "persistence"]),
            r"context": ("上下文", "context", ["state", "session"]),
            
            # 訓練類
            r"train.*|trainer": ("訓練器", "training", ["ml", "ai", "learning"]),
            r"learn.*|learning": ("學習器", "learning", ["ml", "ai", "adaptive"]),
            r"model": ("模型", "model", ["ml", "ai"]),
            
            # 監控類
            r"monitor.*": ("監控", "monitoring", ["observability", "metrics"]),
            r"log.*": ("日誌", "logging", ["observability", "audit"]),
            r"health": ("健康檢查", "health_check", ["monitoring", "diagnostics"]),
            
            # 資料類
            r"payload.*generator": ("載荷生成", "payload_generation", ["attack", "testing"]),
            r"generator": ("生成器", "generator", ["creation", "synthesis"]),
            r"validator": ("驗證器", "validation", ["quality", "check"]),
            r"formatter": ("格式化", "formatting", ["data", "presentation"]),
        }
    
    def _infer_capability_from_path(self, file_path: str, class_name: str) -> Dict[str, Any]:
        """從檔案路徑和類別名稱推斷能力資訊"""
        
        # 提取檔案名（不含副檔名）
        filename = Path(file_path).stem.lower()
        class_lower = class_name.lower()
        combined = f"{filename} {class_lower}"
        
        # 匹配能力模式
        for pattern, (name_zh, name_en, default_tags) in self.capability_patterns.items():
            if re.search(pattern, combined, re.IGNORECASE):
                return {
                    "name_zh": name_zh,
                    "name_en": name_en,
                    "tags": default_tags
                }
        
        # 預設值（無法推斷）
        return {
            "name_zh": class_name,
            "name_en": filename.replace("_", "."),
            "tags": ["utility"]
        }
    
    def _generate_cli_command(self, flow: Dict[str, Any]) -> str:
        """生成 CLI 指令模板"""
        
        full_paths = flow.get("full_path", [])
        if not full_paths:
            return "# 無法生成指令（路徑缺失）"
        
        # 取終點檔案
        end_file = full_paths[-1]
        
        # 提取模組路徑（從 services 開始）
        match = re.search(r'services[/\\].*\.py', end_file.replace('\\', '/'))
        if not match:
            return f"# 無法解析路徑: {end_file}"
        
        module_path = match.group(0).replace('/', '.').replace('\\', '.').replace('.py', '')
        
        # 取終點分類資訊
        classifications = flow.get("classifications", [])
        if classifications:
            last_class = classifications[-1]
            script_name = last_class.get("script", "unknown")
        else:
            script_name = Path(end_file).stem
        
        return f"python -m {module_path} execute"
    
    def _create_capability_info(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """為 Flow 創建能力資訊"""
        
        classifications = flow.get("classifications", [])
        if not classifications:
            return None
        
        # 取終點（最後一個步驟）作為主要能力
        end_point = classifications[-1]
        script_name = end_point.get("script", "unknown")
        module = end_point.get("module", "unknown")
        
        # 從終點路徑推斷能力
        full_paths = flow.get("full_path", [])
        end_file = full_paths[-1] if full_paths else ""
        
        capability = self._infer_capability_from_path(end_file, script_name)
        
        # 構建描述
        module_zh = self.module_names.get(module, module)
        description = f"【{module_zh}】{capability['name_zh']} - 流程長度 {flow.get('length', 0)} 步"
        
        # 添加起點資訊
        if len(classifications) > 1:
            start_point = classifications[0]
            start_script = start_point.get("script", "")
            description += f"，起點：{start_script}"
        
        return {
            "name": capability['name_zh'],
            "identifier": capability['name_en'],
            "description": description,
            "command_template": self._generate_cli_command(flow),
            "tags": capability['tags'] + [module],
            "module": module,
            "module_zh": module_zh,
            "complexity": "simple" if flow.get('length', 0) <= 2 else "medium" if flow.get('length', 0) <= 4 else "complex"
        }
    
    def enrich(self):
        """執行擴充"""
        
        print(f"📖 讀取數據: {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_flows = len(data.get("flows", []))
        print(f"📊 總共 {total_flows} 個 Flows")
        
        enriched_count = 0
        failed_count = 0
        
        for i, flow in enumerate(data.get("flows", []), 1):
            try:
                capability = self._create_capability_info(flow)
                if capability:
                    flow["capability"] = capability
                    enriched_count += 1
                else:
                    failed_count += 1
                    
                # 進度顯示
                if i % 100 == 0:
                    print(f"  處理進度: {i}/{total_flows} ({i*100//total_flows}%)")
                    
            except Exception as e:
                print(f"  ⚠️ Flow {flow.get('id', '?')} 處理失敗: {e}")
                failed_count += 1
        
        # 更新 metadata
        if "metadata" not in data:
            data["metadata"] = {}
        
        data["metadata"]["enriched_at"] = datetime.now().isoformat()
        data["metadata"]["enriched_flows"] = enriched_count
        data["metadata"]["failed_enrichment"] = failed_count
        
        # 輸出
        print(f"\n💾 寫入結果: {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 完成！")
        print(f"   - 成功擴充: {enriched_count} 個")
        print(f"   - 處理失敗: {failed_count} 個")
        print(f"   - 輸出檔案: {self.output_path}")


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description="為 Flows 添加能力描述和 CLI 資訊")
    parser.add_argument(
        '--input',
        default='C:/Users/User/Downloads/data/internal_exploration/latest_classification.json',
        help='輸入 JSON 檔案路徑'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='輸出 JSON 檔案路徑（預設：同目錄的 enriched_classification.json）'
    )
    
    args = parser.parse_args()
    
    enricher = CapabilityEnricher(args.input, args.output)
    enricher.enrich()


if __name__ == "__main__":
    main()
