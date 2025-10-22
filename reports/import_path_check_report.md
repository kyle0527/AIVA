
# AIVA Import Path Checker 報告
生成時間: 2025-10-19 15:51:53

## 摘要
- 檢查檔案總數: 406
- 有問題的檔案數: 18
- 問題總數: 42

## 詳細問題列表

### examples\demo_bio_neuron_master.py
- Line 10: `from aiva_core.bio_neuron_master import (`
  Pattern: `from aiva_core\.`

### services\__init__.py
- Line 40: `import aiva_common`
  Pattern: `import aiva_common\b`

### tools\analyze_aiva_common_status.py
- Line 60: `"from aiva_common.schemas import TaskSchema",`
  Pattern: `from aiva_common\.`
- Line 61: `"from aiva_common.schemas import FindingSchema",`
  Pattern: `from aiva_common\.`
- Line 62: `"from aiva_common.schemas import MessageSchema",`
  Pattern: `from aiva_common\.`
- Line 65: `"from aiva_common.enums import ModuleName",`
  Pattern: `from aiva_common\.`
- Line 66: `"from aiva_common.enums import Severity",`
  Pattern: `from aiva_common\.`
- Line 67: `"from aiva_common.enums import Topic",`
  Pattern: `from aiva_common\.`
- Line 70: `"from aiva_common.schemas.tasks import TaskSchema",`
  Pattern: `from aiva_common\.`
- Line 71: `"from aiva_common.schemas.findings import FindingSchema",`
  Pattern: `from aiva_common\.`
- Line 74: `"from aiva_common.enums.modules import ModuleName",`
  Pattern: `from aiva_common\.`
- Line 75: `"from aiva_common.enums.common import Severity",`
  Pattern: `from aiva_common\.`
- Line 78: `"from aiva_common.schemas import TaskSchema",`
  Pattern: `from aiva_common\.`
- Line 79: `"from aiva_common.enums import TaskStatus",`
  Pattern: `from aiva_common\.`
- Line 123: `if "from aiva_common" in content or "import aiva_common" in content:`
  Pattern: `import aiva_common\b`
- Line 199: `if "from aiva_common.enums import" in init_content:`
  Pattern: `from aiva_common\.`

### tools\create_enums_structure.py
- Line 113: `init_content.append('    from aiva_common.enums import ModuleName, Severity, VulnerabilityType')`
  Pattern: `from aiva_common\.`

### tools\generate_official_schemas.py
- Line 187: `"// AUTO-GENERATED from aiva_common.enums; do not edit.\n",`
  Pattern: `from aiva_common\.`

### tools\import_path_checker.py
- Line 33: `(r'import aiva_core\b', 'import services.core.aiva_core'),`
  Pattern: `import aiva_core\b`
- Line 34: `(r'import aiva_common\b', 'import services.aiva_common'),`
  Pattern: `import aiva_common\b`

### tools\schema_manager.py
- Line 386: `import aiva_common`
  Pattern: `import aiva_common\b`

### tools\schema_validator.py
- Line 240: `import aiva_common`
  Pattern: `import aiva_common\b`

### tools\update_imports.py
- Line 23: `# import aiva_common -> import services.aiva_common`
  Pattern: `import aiva_common\b`
- Line 25: `r"import aiva_common\.", "import services.aiva_common.", content`
  Pattern: `import aiva_common\b`

### tools\verify_migration_completeness.py
- Line 205: `("from aiva_common.enums import ModuleName", "ModuleName"),`
  Pattern: `from aiva_common\.`
- Line 206: `("from aiva_common.enums import Severity", "Severity"),`
  Pattern: `from aiva_common\.`
- Line 207: `("from aiva_common.enums import Topic", "Topic"),`
  Pattern: `from aiva_common\.`
- Line 208: `("from aiva_common.enums import VulnerabilityType", "VulnerabilityType"),`
  Pattern: `from aiva_common\.`
- Line 211: `("from aiva_common.schemas.base import MessageHeader", "MessageHeader"),`
  Pattern: `from aiva_common\.`
- Line 212: `("from aiva_common.schemas.base import Authentication", "Authentication"),`
  Pattern: `from aiva_common\.`

### tools\aiva-enums-plugin\aiva-enums-plugin\scripts\gen_ts_enums.py
- Line 16: `ts_lines: List[str] = ["// AUTO-GENERATED from aiva_common.enums; do not edit.\n\n"]`
  Pattern: `from aiva_common\.`

### services\core\aiva_core\bio_neuron_master.py
- Line 33: `from aiva_core.ai_engine import BioNeuronRAGAgent`
  Pattern: `from aiva_core\.`
- Line 34: `from aiva_core.rag import RAGEngine`
  Pattern: `from aiva_core\.`
- Line 85: `from aiva_core.rag import KnowledgeBase, VectorStore`
  Pattern: `from aiva_core\.`

### services\core\aiva_core\business_schemas.py
- Line 13: `from aiva_common.enums import ModuleName, Severity, TestStatus`
  Pattern: `from aiva_common\.`
- Line 14: `from aiva_common.standards import CVSSv3Metrics`
  Pattern: `from aiva_common\.`

### services\core\aiva_core\__init__.py
- Line 19: `from aiva_common.enums import (`
  Pattern: `from aiva_common\.`
- Line 30: `from aiva_common.schemas import CVEReference, CVSSv3Metrics, CWEReference`
  Pattern: `from aiva_common\.`

### services\core\aiva_core\ai_engine\bio_neuron_core.py
- Line 373: `from aiva_integration.reception.experience_repository import (`
  Pattern: `from aiva_integration\.`

### services\core\aiva_core\rag\demo_rag_integration.py
- Line 13: `from aiva_core.rag import KnowledgeBase, RAGEngine, VectorStore`
  Pattern: `from aiva_core\.`

### services\aiva_common\enums\__init__.py
- Line 7: `from aiva_common.enums import ModuleName, Severity, VulnerabilityType`
  Pattern: `from aiva_common\.`

### services\aiva_common\schemas\__init__.py
- Line 7: `from aiva_common.schemas import FindingPayload, ScanStartPayload, MessageHeader`
  Pattern: `from aiva_common\.`

## 建議修復命令
```bash
python tools/import_path_checker.py --fix
```

## 預防措施
1. 在 pre-commit hook 中加入此檢查
2. 在 CI/CD pipeline 中加入自動檢查
3. 定期執行完整掃描
