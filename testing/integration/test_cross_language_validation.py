#!/usr/bin/env python3
import sys
import os

# 設置路徑和環境變數
sys.path.insert(0, 'services/aiva_common/tools')
# v2.0 架構：無需設置外部服務環境變數
# 所有通信通過數據合約，所有存儲使用本地檔案

from cross_language_validator import CrossLanguageValidator

def main():
    validator = CrossLanguageValidator()
    report = validator.validate_all()
    
    # 打印摘要
    validator.print_summary(report)
    
    # 生成報告文件
    validator.generate_report_file(report, "cross_language_validation_report.json")
    
    # 檢查是否有嚴重問題
    critical_issues = len([i for i in report.issues if i.severity == 'critical'])
    success = critical_issues == 0
    
    print(f'\n🎯 最終結果: {"✅ 驗證通過" if success else f"❌ {critical_issues} 個嚴重問題需要修復"}')
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)