"""
分析 schemas.py 中尚未遷移的類別
"""

import re
from pathlib import Path

schemas_file = Path('schemas.py')
content = schemas_file.read_text(encoding='utf-8')

# 提取所有類別定義和其上下文
class_pattern = re.compile(r'^((?:# .*\n)*class\s+(\w+)\s*\([^)]+\):[^\n]*(?:\n    [^\n]*)*)', re.MULTILINE)

missing_classes = [
    'AIExperienceCreatedEvent',
    'AIModelDeployCommand',
    'AIModelUpdatedEvent',
    'AITraceCompletedEvent',
    'AITrainingCompletedPayload',
    'AssetInventoryItem',
    'CVEReference',
    'CWEReference',
    'EASMAsset',
    'EnhancedAttackPath',
    'EnhancedAttackPathNode',
    'EnhancedFindingPayload',
    'EnhancedFunctionTaskTarget',
    'EnhancedIOCRecord',
    'EnhancedModuleStatus',
    'EnhancedRiskAssessment',
    'EnhancedScanRequest',
    'EnhancedScanScope',
    'EnhancedTaskExecution',
    'EnhancedVulnerabilityCorrelation',
    'ExploitPayload',
    'ExploitResult',
    'ModelTrainingResult',
    'RAGResponsePayload',
    'SIEMEvent',
    'ScenarioTestResult',
    'SessionState',
    'StandardScenario',
    'SystemOrchestration',
    'TaskQueue',
    'TechnicalFingerprint',
    'TestExecution',
    'TestStrategy',
    'VulnerabilityDiscovery',
    'WebhookPayload',
]

# 分類
categories = {
    'AI相關': [],
    'Enhanced版本': [],
    'Event/Command': [],
    'EASM相關': [],
    '測試相關': [],
    '其他': []
}

for cls in missing_classes:
    if 'AI' in cls and ('Event' in cls or 'Command' in cls or 'Payload' in cls or 'Result' in cls):
        categories['AI相關'].append(cls)
    elif cls.startswith('Enhanced'):
        categories['Enhanced版本'].append(cls)
    elif 'Event' in cls or 'Command' in cls:
        categories['Event/Command'].append(cls)
    elif 'EASM' in cls or 'Asset' in cls:
        categories['EASM相關'].append(cls)
    elif 'Test' in cls or 'Scenario' in cls or 'Exploit' in cls:
        categories['測試相關'].append(cls)
    else:
        categories['其他'].append(cls)

print("=" * 70)
print("未遷移類別分類報告")
print("=" * 70)

for category, classes in categories.items():
    if classes:
        print(f"\n{category} ({len(classes)} 個):")
        for cls in sorted(classes):
            print(f"  - {cls}")

# 建議遷移到的檔案
print("\n" + "=" * 70)
print("建議遷移映射")
print("=" * 70)

migrations = {
    'schemas/ai.py': categories['AI相關'],
    'schemas/messaging.py': categories['Event/Command'],
    'schemas/assets.py': categories['EASM相關'],
    'schemas/tasks.py': categories['測試相關'],
    'schemas/enhanced.py (新建)': categories['Enhanced版本'],
    'schemas/misc.py (新建)': categories['其他']
}

for target, classes in migrations.items():
    if classes:
        print(f"\n{target}:")
        for cls in sorted(classes):
            print(f"  - {cls}")

print(f"\n總計: {len(missing_classes)} 個類別待遷移")
