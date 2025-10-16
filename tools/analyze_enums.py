"""
分析 enums.py 並按功能分類
"""

enums_classification = {
    'common.py': {
        'description': '通用枚舉 - 狀態、級別、類型等基礎枚舉',
        'enums': [
            'Severity',           # 嚴重性
            'Confidence',         # 置信度
            'TaskStatus',         # 任務狀態
            'TestStatus',         # 測試狀態
            'ScanStatus',         # 掃描狀態
            'ThreatLevel',        # 威脅級別
            'RiskLevel',          # 風險級別
            'RemediationStatus',  # 修復狀態
        ]
    },
    
    'modules.py': {
        'description': '模組相關枚舉 - 模組名稱、主題等',
        'enums': [
            'ModuleName',         # 模組名稱
            'Topic',              # 消息主題
        ]
    },
    
    'security.py': {
        'description': '安全測試相關枚舉 - 漏洞、攻擊、權限等',
        'enums': [
            'VulnerabilityType',      # 漏洞類型
            'VulnerabilityStatus',    # 漏洞狀態
            'Location',               # 參數位置
            'SensitiveInfoType',      # 敏感信息類型
            'IntelSource',            # 情報來源
            'IOCType',                # IOC 類型
            'RemediationType',        # 修復類型
            'Permission',             # 權限
            'AccessDecision',         # 訪問決策
            'PostExTestType',         # 後滲透測試類型
            'PersistenceType',        # 持久化類型
            'Exploitability',         # 可利用性
            'AttackPathNodeType',     # 攻擊路徑節點類型
            'AttackPathEdgeType',     # 攻擊路徑邊類型
        ]
    },
    
    'assets.py': {
        'description': '資產管理相關枚舉 - 資產類型、環境、合規等',
        'enums': [
            'BusinessCriticality',    # 業務重要性
            'Environment',            # 環境類型
            'AssetType',              # 資產類型
            'AssetStatus',            # 資產狀態
            'DataSensitivity',        # 資料敏感度
            'AssetExposure',          # 資產暴露度
            'ComplianceFramework',    # 合規框架
        ]
    }
}

# 統計
print("=" * 70)
print("Enums 分類計劃")
print("=" * 70)

total = 0
for filename, info in enums_classification.items():
    count = len(info['enums'])
    total += count
    print(f"\n📄 {filename} ({count} 個枚舉)")
    print(f"   {info['description']}")
    for enum in info['enums']:
        print(f"   - {enum}")

print(f"\n{'=' * 70}")
print(f"總計: {total} 個枚舉")
print("=" * 70)

# 檢查是否有遺漏
all_enums = [
    'ModuleName', 'Topic', 'Severity', 'Confidence', 'VulnerabilityType',
    'TaskStatus', 'TestStatus', 'ScanStatus', 'SensitiveInfoType', 'Location',
    'ThreatLevel', 'IntelSource', 'IOCType', 'RemediationType', 'RemediationStatus',
    'Permission', 'AccessDecision', 'PostExTestType', 'PersistenceType',
    'BusinessCriticality', 'Environment', 'AssetType', 'AssetStatus',
    'VulnerabilityStatus', 'DataSensitivity', 'AssetExposure', 'Exploitability',
    'ComplianceFramework', 'RiskLevel', 'AttackPathNodeType', 'AttackPathEdgeType'
]

classified = []
for info in enums_classification.values():
    classified.extend(info['enums'])

missing = set(all_enums) - set(classified)
if missing:
    print(f"\n⚠️  未分類的枚舉: {missing}")
else:
    print(f"\n✅ 所有 {len(all_enums)} 個枚舉都已分類")

# 生成檔案結構
print("\n" + "=" * 70)
print("目標資料夾結構:")
print("=" * 70)
print("""
enums/
├── __init__.py       # 統一導出所有枚舉
├── common.py         # 8 個通用枚舉
├── modules.py        # 2 個模組相關枚舉
├── security.py       # 14 個安全測試枚舉
└── assets.py         # 7 個資產管理枚舉
""")
