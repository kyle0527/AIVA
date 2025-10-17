"""
🎯 一鍵安全分析 - All-in-One Security Analysis
整合所有安全分析工具的便捷腳本

功能:
1. 日誌分析 (Log Analysis)
2. 攻擊模式訓練 (Attack Pattern Training)  
3. 威脅檢測報告 (Threat Detection)
4. 生成完整安全報告
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd: list, description: str) -> bool:
    """執行命令"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - 失敗: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ 找不到 Python 或腳本文件")
        return False

def main():
    """主程序"""
    print("=" * 70)
    print("🎯 一鍵安全分析 - All-in-One Security Analysis")
    print("=" * 70)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    log_file = "AI_OPTIMIZATION_REQUIREMENTS.txt"
    
    # 檢查日誌文件
    if not Path(log_file).exists():
        print(f"\n❌ 錯誤: 找不到日誌文件 {log_file}")
        return
    
    print(f"\n📁 日誌文件: {log_file}")
    print(f"📊 文件大小: {Path(log_file).stat().st_size / 1024:.2f} KB")
    
    # 步驟 1: 分析日誌
    success = run_command(
        [sys.executable, "tools/security_log_analyzer.py", log_file],
        "步驟 1/3: 安全日誌分析"
    )
    
    if not success:
        print("\n⚠️  日誌分析失敗,繼續下一步...")
    
    # 步驟 2: 訓練攻擊模式
    success = run_command(
        [sys.executable, "tools/attack_pattern_trainer.py", 
         "--epochs", "100", 
         "--data", "_out/attack_training_data.json"],
        "步驟 2/3: 攻擊模式訓練"
    )
    
    if not success:
        print("\n⚠️  模式訓練失敗,繼續下一步...")
    
    # 步驟 3: 威脅檢測
    success = run_command(
        [sys.executable, "tools/real_time_threat_detector.py", log_file,
         "--threshold", "3"],
        "步驟 3/3: 威脅檢測分析"
    )
    
    # 最終報告
    print("\n" + "=" * 70)
    print("📊 分析完成摘要")
    print("=" * 70)
    
    output_files = [
        ("安全分析報告", "_out/security_analysis_report.md"),
        ("訓練數據", "_out/attack_training_data.json"),
        ("檢測模型", "_out/attack_detection_model.json"),
        ("威脅報告", "_out/threat_detection_report.json")
    ]
    
    print("\n生成的文件:")
    for name, path in output_files:
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024
            print(f"  ✓ {name}: {path} ({size:.2f} KB)")
        else:
            print(f"  ✗ {name}: {path} (未生成)")
    
    print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("✓ 所有分析完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
