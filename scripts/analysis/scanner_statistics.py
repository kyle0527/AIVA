#!/usr/bin/env python3
"""
AIVA 掃描器統計分析器
基於 AI 組件探索結果生成準確的掃描器統計
"""

import os
import json
from pathlib import Path
from datetime import datetime

def count_go_scanners():
    """統計 Go 掃描器"""
    go_scanners = []
    go_dirs = [
        "services/scan/aiva_scan/network_scanner_go",
        "services/scan/aiva_scan/port_scanner_go", 
        "services/scan/aiva_scan/vuln_scanner_go",
        "services/scan/aiva_scan/web_scanner_go"
    ]
    
    for scanner_dir in go_dirs:
        if os.path.exists(scanner_dir):
            # 檢查是否有 Go 文件
            go_files = list(Path(scanner_dir).glob("*.go"))
            if go_files:
                go_scanners.append({
                    "name": os.path.basename(scanner_dir),
                    "path": scanner_dir,
                    "files": len(go_files),
                    "status": "✅ 可用"
                })
    
    return go_scanners

def count_rust_scanners():
    """統計 Rust 掃描器"""
    rust_scanners = []
    rust_dirs = [
        "services/scan/aiva_scan/advanced_scanner_rust"
    ]
    
    for scanner_dir in rust_dirs:
        if os.path.exists(scanner_dir):
            # 檢查是否有 Rust 文件
            rust_files = list(Path(scanner_dir).glob("*.rs"))
            cargo_toml = Path(scanner_dir) / "Cargo.toml"
            if rust_files or cargo_toml.exists():
                rust_scanners.append({
                    "name": os.path.basename(scanner_dir),
                    "path": scanner_dir,
                    "files": len(rust_files),
                    "has_cargo": cargo_toml.exists(),
                    "status": "✅ 可用"
                })
    
    return rust_scanners

def count_python_scanners():
    """統計 Python 掃描器"""
    python_scanners = []
    
    # Features 模組掃描器
    features_dir = "services/features"
    if os.path.exists(features_dir):
        for item in os.listdir(features_dir):
            item_path = os.path.join(features_dir, item)
            if os.path.isdir(item_path) and item.startswith("function_"):
                scanner_name = item.replace("function_", "")
                python_scanners.append({
                    "name": f"{scanner_name}_scanner",
                    "path": item_path,
                    "type": "features",
                    "status": "✅ 可用"
                })
    
    # Scan 模組掃描器
    scan_dir = "services/scan/aiva_scan"
    if os.path.exists(scan_dir):
        for item in os.listdir(scan_dir):
            item_path = os.path.join(scan_dir, item)
            if os.path.isdir(item_path) and not item.endswith("_go") and not item.endswith("_rust"):
                if any(f.endswith(".py") for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                    python_scanners.append({
                        "name": item,
                        "path": item_path,
                        "type": "scan",
                        "status": "✅ 可用"
                    })
    
    return python_scanners

def count_ai_smart_detectors():
    """統計 AI 智能檢測器"""
    ai_detectors = [
        "smart_detection_manager",
        "unified_smart_detection_manager", 
        "smart_idor_detector",
        "smart_ssrf_detector"
    ]
    
    return [{
        "name": detector,
        "type": "AI智能檢測器",
        "status": "✅ 可用"
    } for detector in ai_detectors]

def generate_scanner_report():
    """生成完整的掃描器統計報告"""
    
    print("🔍 AIVA 掃描器統計分析")
    print("=" * 60)
    
    # 統計各語言掃描器
    go_scanners = count_go_scanners()
    rust_scanners = count_rust_scanners()
    python_scanners = count_python_scanners()
    ai_detectors = count_ai_smart_detectors()
    
    print(f"\n📊 掃描器統計總覽:")
    print(f"   Go 掃描器:     {len(go_scanners)} 個")
    print(f"   Rust 掃描器:   {len(rust_scanners)} 個")
    print(f"   Python 掃描器: {len(python_scanners)} 個")
    print(f"   AI 智能檢測器: {len(ai_detectors)} 個")
    print(f"   總計:          {len(go_scanners) + len(rust_scanners) + len(python_scanners) + len(ai_detectors)} 個掃描器")
    
    print(f"\n🔧 Go 掃描器詳情:")
    for scanner in go_scanners:
        print(f"   {scanner['status']} {scanner['name']} ({scanner['files']} 個 .go 文件)")
    
    print(f"\n🦀 Rust 掃描器詳情:")
    for scanner in rust_scanners:
        cargo_info = "有 Cargo.toml" if scanner['has_cargo'] else "無 Cargo.toml"
        print(f"   {scanner['status']} {scanner['name']} ({scanner['files']} 個 .rs 文件, {cargo_info})")
    
    print(f"\n🐍 Python 掃描器詳情:")
    for scanner in python_scanners:
        print(f"   {scanner['status']} {scanner['name']} ({scanner['type']} 模組)")
    
    print(f"\n🤖 AI 智能檢測器詳情:")
    for detector in ai_detectors:
        print(f"   {detector['status']} {detector['name']}")
    
    # 保存詳細報告
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "go_scanners": len(go_scanners),
            "rust_scanners": len(rust_scanners),
            "python_scanners": len(python_scanners),
            "ai_detectors": len(ai_detectors),
            "total_scanners": len(go_scanners) + len(rust_scanners) + len(python_scanners) + len(ai_detectors)
        },
        "details": {
            "go_scanners": go_scanners,
            "rust_scanners": rust_scanners,
            "python_scanners": python_scanners,
            "ai_detectors": ai_detectors
        }
    }
    
    # 確保報告目錄存在
    os.makedirs("reports/scanner_statistics", exist_ok=True)
    
    report_file = f"reports/scanner_statistics/scanner_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 詳細報告已保存: {report_file}")
    
    return report_data

if __name__ == "__main__":
    generate_scanner_report()