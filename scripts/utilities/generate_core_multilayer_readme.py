#!/usr/bin/env python3
"""
AIVA Core 模組多層次 README 生成器
基於 generate_multilayer_readme.py 為 Core 模組創建完整的文件架構
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

class CoreMultiLayerReadmeGenerator:
    """Core 模組多層次 README 生成器"""
    
    def __init__(self):
        self.base_dir = Path("services/core")
        self.output_dir = self.base_dir / "docs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入分析資料
        analysis_file = Path("_out/core_module_analysis_detailed.json")
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                self.analysis_data = json.load(f)
        else:
            print("⚠️ 找不到分析數據，請先運行 analyze_core_modules.py")
            self.analysis_data = []
        
        # 統計數據
        self.stats = self._calculate_statistics()
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """計算統計數據"""
        if not self.analysis_data:
            return {}
        
        total_files = len(self.analysis_data)
        total_code_lines = sum(item['code_lines'] for item in self.analysis_data)
        total_classes = sum(item['classes'] for item in self.analysis_data)
        total_functions = sum(item['functions'] for item in self.analysis_data)
        total_async = sum(item['async_functions'] for item in self.analysis_data)
        avg_complexity = sum(item['complexity_score'] for item in self.analysis_data) / total_files
        
        # 按功能分類
        ai_files = [f for f in self.analysis_data if 'ai_' in f['file'] or 'bio_neuron' in f['file'] or 'nlg' in f['file']]
        execution_files = [f for f in self.analysis_data if any(k in f['file'] for k in ['execution', 'task_', 'plan_'])]
        analysis_files = [f for f in self.analysis_data if 'analysis' in f['file'] or 'decision' in f['file']]
        storage_files = [f for f in self.analysis_data if any(k in f['file'] for k in ['storage', 'state', 'session'])]
        learning_files = [f for f in self.analysis_data if any(k in f['file'] for k in ['learning', 'training'])]
        
        # 依賴分析
        all_imports = defaultdict(int)
        for item in self.analysis_data:
            for module in item.get('import_modules', []):
                if module and not module.startswith('.'):
                    all_imports[module] += 1
        
        return {
            'total_files': total_files,
            'total_code_lines': total_code_lines,
            'total_classes': total_classes,
            'total_functions': total_functions,
            'total_async_functions': total_async,
            'avg_complexity': round(avg_complexity, 1),
            'ai_components': len(ai_files),
            'execution_components': len(execution_files),
            'analysis_components': len(analysis_files),
            'storage_components': len(storage_files),
            'learning_components': len(learning_files),
            'top_dependencies': sorted(all_imports.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def generate_main_readme(self) -> str:
        """生成主 README - 架構總覽與導航"""
        
        stats = self.stats
        
        template = f"""# AIVA Core 模組 - AI驅動核心引擎架構

> **🎯 快速導航**: 選擇您的角色和需求，找到最適合的文件
> 
> - 👨‍💼 **架構師/PM**: 閱讀 [核心架構總覽](#核心架構總覽)
> - 🐍 **Python 開發者**: 查看 [開發指南](docs/README_DEVELOPMENT.md)
> - 🤖 **AI 工程師**: 查看 [AI 引擎指南](docs/README_AI_ENGINE.md)
> - ⚡ **性能工程師**: 查看 [執行引擎指南](docs/README_EXECUTION.md)
> - 🧠 **ML 工程師**: 查看 [學習系統指南](docs/README_LEARNING.md)

---

## 📊 **模組規模一覽**

### **🏗️ 整體統計**
- **總檔案數**: **{stats['total_files']}** 個 Python 模組
- **代碼行數**: **{stats['total_code_lines']:,}** 行
- **類別數量**: **{stats['total_classes']}** 個類別
- **函數數量**: **{stats['total_functions']}** 個函數 (含 {stats['total_async_functions']} 個異步函數)
- **平均複雜度**: **{stats['avg_complexity']}** / 100
- **複雜度等級**: ⭐⭐⭐⭐⭐ (最高級別)

### **📈 功能分佈**
```
🤖 AI 引擎        │████████████████████████████████████ {stats['ai_components']} 組件
⚡ 執行引擎        │██████████████████████ {stats['execution_components']} 組件
🧠 學習系統        │████████████ {stats['learning_components']} 組件
📊 分析決策        │██████████ {stats['analysis_components']} 組件
💾 存儲狀態        │████████ {stats['storage_components']} 組件
```

---

## 🏗️ **核心架構總覽**

### **五層核心架構**

```mermaid
flowchart TD
    CORE["🎯 AIVA Core Engine<br/>{stats['total_files']} 組件"]
    
    AI["🤖 AI 引擎層<br/>{stats['ai_components']} 組件<br/>智能決策與控制"]
    EXEC["⚡ 執行引擎層<br/>{stats['execution_components']} 組件<br/>任務調度與執行"]
    LEARN["🧠 學習系統層<br/>{stats['learning_components']} 組件<br/>持續學習與優化"]
    ANALYSIS["📊 分析決策層<br/>{stats['analysis_components']} 組件<br/>風險評估與策略"]
    STORAGE["💾 存儲管理層<br/>{stats['storage_components']} 組件<br/>狀態與數據管理"]
    
    CORE --> AI
    CORE --> EXEC
    CORE --> LEARN
    CORE --> ANALYSIS
    CORE --> STORAGE
    
    AI <--> EXEC
    EXEC <--> LEARN
    LEARN <--> ANALYSIS
    ANALYSIS <--> STORAGE
    
    classDef aiStyle fill:#9333ea,color:#fff
    classDef execStyle fill:#dc2626,color:#fff
    classDef learnStyle fill:#2563eb,color:#fff
    classDef analysisStyle fill:#059669,color:#fff
    classDef storageStyle fill:#ea580c,color:#fff
    
    class AI aiStyle
    class EXEC execStyle
    class LEARN learnStyle
    class ANALYSIS analysisStyle
    class STORAGE storageStyle
```

### **🎯 各層核心職責**

| 功能層 | 主要職責 | 關鍵模組 | 代碼規模 |
|--------|----------|----------|----------|
| 🤖 **AI 引擎** | AI模型管理、神經網絡、反幻覺 | bio_neuron_core, ai_controller | 2,000+ 行 |
| ⚡ **執行引擎** | 任務調度、計劃執行、狀態監控 | plan_executor, task_dispatcher | 1,500+ 行 |
| 🧠 **學習系統** | 模型訓練、經驗管理、場景訓練 | model_trainer, scenario_manager | 1,700+ 行 |
| 📊 **分析決策** | 風險評估、策略生成、決策支持 | enhanced_decision_agent, strategy_generator | 800+ 行 |
| 💾 **存儲管理** | 狀態管理、數據持久化、會話管理 | session_state_manager, storage_manager | 600+ 行 |

---

## 📚 **文件導航地圖**

### **📁 按功能查看**
- 🤖 [**AI 引擎詳解**](docs/README_AI_ENGINE.md) - 生物神經網絡、AI控制器、反幻覺模組
- ⚡ [**執行引擎詳解**](docs/README_EXECUTION.md) - 任務調度、計劃執行、監控追蹤
- 🧠 [**學習系統詳解**](docs/README_LEARNING.md) - 模型訓練、經驗管理、場景訓練
- 📊 [**分析決策詳解**](docs/README_ANALYSIS.md) - 風險評估、策略生成、決策代理
- 💾 [**存儲管理詳解**](docs/README_STORAGE.md) - 狀態管理、數據持久化、會話控制

### **💻 開發文檔**
- 🐍 [**開發指南**](docs/README_DEVELOPMENT.md) - Python 開發規範、最佳實踐
- 🔧 [**API 參考**](docs/README_API.md) - 核心 API 文檔與使用範例
- 🧪 [**測試指南**](docs/README_TESTING.md) - 單元測試、整合測試策略

---

## 🚀 **快速開始指南**

### **🔍 我需要什麼？**

**場景 1: 了解 AI 引擎** 🤖  
```
→ 閱讀本文件的核心架構總覽
→ 查看 docs/README_AI_ENGINE.md
→ 檢視 bio_neuron_core.py 和 ai_controller.py
```

**場景 2: 開發任務執行功能** ⚡  
```
→ 閱讀 docs/README_EXECUTION.md
→ 查看 plan_executor.py 和 task_dispatcher.py
→ 跟隨執行引擎開發模式
```

**場景 3: 實現學習功能** 🧠  
```  
→ 閱讀 docs/README_LEARNING.md
→ 查看 model_trainer.py 和 scenario_manager.py
→ 跟隨學習系統開發指南
```

**場景 4: 系統整合與部署** 🔧  
```
→ 閱讀 docs/README_DEVELOPMENT.md  
→ 查看整合測試範例
→ 參考部署和監控最佳實踐
```

### **🛠️ 環境設定**
```bash
# 1. 進入 Core 模組
cd services/core

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 配置環境變量
cp .env.example .env

# 4. 執行測試
python -m pytest tests/ -v

# 5. 啟動開發服務器
python -m aiva_core.app
```

---

## ⚠️ **重要注意事項**

### **🔴 關鍵架構原則**
1. **AI 優先**: Core 模組以 AI 引擎為核心
2. **異步優先**: 大量使用異步編程提升性能
3. **狀態管理**: 嚴格的狀態管理和持久化策略
4. **模組化設計**: 清晰的層次結構和依賴關係

### **🚨 開發約束**
- ✅ **必須**: 遵循 Python 類型標註和文檔字符串規範
- ✅ **必須**: 實現完整的錯誤處理和日誌記錄
- ⚠️ **避免**: 跨層直接調用，應通過定義的介面
- ⚠️ **避免**: 阻塞操作，優先使用異步方法

---

## 📈 **技術債務與優化建議**

### **🚨 高複雜度模組 (需要重構)**
基於代碼分析，以下模組複雜度較高，建議優先重構：

1. **bio_neuron_core.py** (複雜度: 97)
   - 建議拆分為多個專門模組
   - 最長函數 118 行，需要分解

2. **ai_controller.py** (複雜度: 77)
   - 統一控制器邏輯過於龐大
   - 建議引入更多委託模式

3. **enhanced_decision_agent.py** (複雜度: 75)
   - 決策邏輯複雜度高
   - 建議引入策略模式簡化

### **⚡ 性能優化機會**
- 增加異步函數使用率（當前 {stats['total_async_functions']} / {stats['total_functions']}）
- 實現更完善的緩存策略
- 優化數據庫查詢和批量操作

---

## 🔗 **核心依賴關係**

### **📦 主要外部依賴**
{self._format_dependencies()}

---

## 📞 **支援與聯繫**

### **👥 團隊分工**
- 🤖 **AI 引擎團隊**: 神經網絡、模型管理
- ⚡ **執行引擎團隊**: 任務調度、性能優化
- 🧠 **學習系統團隊**: 訓練管道、經驗管理
- 📊 **分析團隊**: 決策系統、風險評估

### **📊 相關報告**
- 📈 [核心模組代碼分析](_out/core_module_analysis_detailed.json)
- 🔍 [架構優化建議](reports/ANALYSIS_REPORTS/core_module_comprehensive_analysis.md)

---

**📝 文件版本**: v1.0 - Core Module Multi-Layer Documentation  
**🔄 最後更新**: {datetime.now().strftime('%Y-%m-%d')}  
**📈 複雜度等級**: ⭐⭐⭐⭐⭐ (最高) - 核心引擎系統  
**👥 維護團隊**: AIVA Core Architecture Team

*這是 AIVA Core 模組的主要導航文件。根據您的角色和需求，選擇適合的專業文件深入了解。*
"""
        return template
    
    def _format_dependencies(self) -> str:
        """格式化依賴列表"""
        deps = self.stats.get('top_dependencies', [])
        lines = []
        for module, count in deps:
            lines.append(f"- **{module}**: {count} 次引用")
        return '\n'.join(lines)
    
    def generate_ai_engine_readme(self) -> str:
        """生成 AI 引擎專門 README"""
        
        ai_files = [f for f in self.analysis_data if 'ai_' in f['file'] or 'bio_neuron' in f['file'] or 'nlg' in f['file']]
        
        template = f"""# AIVA Core - AI 引擎架構詳解 🤖

> **定位**: AIVA 平台的 AI 核心引擎  
> **規模**: {len(ai_files)} 個 AI 組件  
> **主力技術**: 生物神經網絡、反幻覺系統、自然語言生成

---

## 🎯 **AI 引擎總覽**

### **🔥 AI 引擎架構**

```
🤖 AI 引擎層
├── 🧠 生物神經網絡 (bio_neuron_core.py)
│   ├── 生物脈衝層 (BiologicalSpikingLayer)
│   ├── 反幻覺模組 (AntiHallucinationModule)
│   └── 可擴展生物網絡 (ScalableBioNet)
├── 🎛️ AI 控制器 (ai_controller.py)
│   ├── 統一 AI 控制器 (UnifiedAIController)
│   └── 多語言協調整合
├── 🧩 AI 指揮官 (ai_commander.py)
│   ├── AI 任務類型管理
│   ├── AI 組件協調
│   └── 狀態管理與保存
├── 🧠 AI 模型管理器 (ai_model_manager.py)
│   └── 模型生命週期管理
├── 💬 自然語言生成 (nlg_system.py)
│   └── 智能文本生成系統
└── 🔌 AI 摘要插件 (ai_summary_plugin.py)
    └── 智能摘要功能
```

### **⚡ 核心能力**

| AI 模組 | 主要功能 | 代碼規模 | 複雜度 |
|---------|----------|----------|--------|
| **bio_neuron_core** | 生物神經網絡、反幻覺 | 648 行 | 97 |
| **ai_controller** | 統一 AI 控制 | 621 行 | 77 |
| **bio_neuron_master** | 主控制器 | 488 行 | 45 |
| **ai_model_manager** | 模型管理 | 370 行 | 38 |
| **nlg_system** | 自然語言生成 | 365 行 | 43 |

---

## 🧠 **生物神經網絡核心**

### **核心架構**

```python
from typing import Dict, List, Optional
import torch
import torch.nn as nn

class BiologicalSpikingLayer(nn.Module):
    \"\"\"生物脈衝神經網絡層\"\"\"
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"前向傳播 - 生物脈衝機制\"\"\"
        # 實現生物脈衝邏輯
        return self._biological_spike(x)
    
    def _biological_spike(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"生物脈衝計算\"\"\"
        # 模擬神經元脈衝行為
        pass

class AntiHallucinationModule(nn.Module):
    \"\"\"反幻覺模組 - 確保 AI 輸出可靠性\"\"\"
    
    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        
    def validate_output(self, output: Dict, context: Dict) -> bool:
        \"\"\"驗證輸出是否可靠\"\"\"
        confidence = self._calculate_confidence(output, context)
        return confidence >= self.confidence_threshold
    
    def _calculate_confidence(self, output: Dict, context: Dict) -> float:
        \"\"\"計算輸出信心度\"\"\"
        # 多維度信心度評估
        pass

class ScalableBioNet(nn.Module):
    \"\"\"可擴展生物神經網絡\"\"\"
    
    def __init__(self, config: Dict):
        super().__init__()
        self.layers = nn.ModuleList([
            BiologicalSpikingLayer(config['input_size'], config['hidden_size']),
            BiologicalSpikingLayer(config['hidden_size'], config['output_size'])
        ])
        self.anti_hallucination = AntiHallucinationModule()
    
    async def forward_with_validation(self, x: torch.Tensor, context: Dict) -> Dict:
        \"\"\"帶驗證的前向傳播\"\"\"
        output = self.forward(x)
        
        # 反幻覺驗證
        is_valid = self.anti_hallucination.validate_output(output, context)
        
        return {{
            'output': output,
            'valid': is_valid,
            'confidence': self.anti_hallucination._calculate_confidence(output, context)
        }}
```

---

## 🎛️ **統一 AI 控制器**

### **控制器架構**

```python
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class AIControllerConfig:
    \"\"\"AI 控制器配置\"\"\"
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    use_anti_hallucination: bool = True
    enable_learning: bool = True

class UnifiedAIController:
    \"\"\"統一 AI 控制器 - 協調所有 AI 組件\"\"\"
    
    def __init__(self, config: AIControllerConfig):
        self.config = config
        self.bio_net = ScalableBioNet(self._get_bio_net_config())
        self.nlg_system = AIVANaturalLanguageGenerator()
        self.model_manager = AIModelManager()
        
    async def process_request(self, request: Dict) -> Dict:
        \"\"\"處理 AI 請求 - 主要入口點\"\"\"
        
        # 1. 預處理
        processed_input = await self._preprocess_request(request)
        
        # 2. AI 推理
        raw_output = await self._run_inference(processed_input)
        
        # 3. 反幻覺驗證
        if self.config.use_anti_hallucination:
            validated_output = await self._validate_output(raw_output, request)
        else:
            validated_output = raw_output
        
        # 4. 自然語言生成
        final_response = await self._generate_response(validated_output)
        
        # 5. 學習與更新
        if self.config.enable_learning:
            await self._update_learning(request, final_response)
        
        return final_response
    
    async def _run_inference(self, input_data: Dict) -> Dict:
        \"\"\"執行 AI 推理\"\"\"
        # 使用生物神經網絡進行推理
        tensor_input = self._convert_to_tensor(input_data)
        result = await self.bio_net.forward_with_validation(
            tensor_input, 
            context=input_data.get('context', {{}})
        )
        return result
    
    async def _validate_output(self, output: Dict, original_request: Dict) -> Dict:
        \"\"\"驗證輸出 - 防止幻覺\"\"\"
        if not output.get('valid', False):
            # 輸出不可靠，重新生成或使用備用策略
            return await self._fallback_generation(original_request)
        return output
```

---

## 💬 **自然語言生成系統**

### **NLG 架構**

```python
class AIVANaturalLanguageGenerator:
    \"\"\"AIVA 自然語言生成器\"\"\"
    
    def __init__(self):
        self.templates = self._load_templates()
        self.context_manager = ContextManager()
        
    async def generate(self, data: Dict, style: str = "professional") -> str:
        \"\"\"生成自然語言輸出\"\"\"
        
        # 1. 選擇模板
        template = self._select_template(data['type'], style)
        
        # 2. 填充上下文
        context = await self.context_manager.build_context(data)
        
        # 3. 生成文本
        generated_text = self._fill_template(template, context)
        
        # 4. 後處理
        polished_text = self._polish_text(generated_text)
        
        return polished_text
    
    def _select_template(self, data_type: str, style: str) -> str:
        \"\"\"選擇合適的模板\"\"\"
        key = f"{{data_type}}_{{style}}"
        return self.templates.get(key, self.templates['default'])
    
    def _polish_text(self, text: str) -> str:
        \"\"\"文本潤色\"\"\"
        # 語法檢查、格式化、優化可讀性
        pass
```

---

## 🧪 **測試與驗證**

### **AI 引擎測試**

```python
import pytest
import asyncio

class TestBioNeuronCore:
    
    async def test_biological_spike_layer(self):
        \"\"\"測試生物脈衝層\"\"\"
        layer = BiologicalSpikingLayer(input_size=10, output_size=5)
        input_tensor = torch.randn(1, 10)
        
        output = layer(input_tensor)
        
        assert output.shape == (1, 5)
        assert torch.all(torch.isfinite(output))
    
    async def test_anti_hallucination(self):
        \"\"\"測試反幻覺模組\"\"\"
        module = AntiHallucinationModule(confidence_threshold=0.7)
        
        # 高信心度輸出
        valid_output = {{'data': 'test', 'confidence': 0.85}}
        assert module.validate_output(valid_output, {{}}) == True
        
        # 低信心度輸出
        invalid_output = {{'data': 'test', 'confidence': 0.5}}
        assert module.validate_output(invalid_output, {{}}) == False

@pytest.mark.asyncio
class TestUnifiedAIController:
    
    async def test_process_request(self):
        \"\"\"測試 AI 請求處理\"\"\"
        config = AIControllerConfig(model_name="bio-gpt")
        controller = UnifiedAIController(config)
        
        request = {{
            'type': 'scan_analysis',
            'data': {{'target': 'example.com'}},
            'context': {{'user': 'test_user'}}
        }}
        
        response = await controller.process_request(request)
        
        assert 'output' in response
        assert response.get('valid', False) == True
```

---

**📝 版本**: v1.0 - AI Engine Deep Dive  
**🔄 最後更新**: {datetime.now().strftime('%Y-%m-%d')}  
**🤖 AI 技術棧**: PyTorch + 生物神經網絡 + 反幻覺系統  
**👥 維護團隊**: AIVA AI Engine Team

*本文件詳細介紹 AIVA Core 模組的 AI 引擎架構，包含生物神經網絡、AI 控制器和自然語言生成系統。*
"""
        return template
    
    def run_generation(self):
        """執行 README 生成"""
        print("🚀 開始生成 Core 模組多層次 README 架構...")
        
        readmes = {
            "README.md": self.generate_main_readme(),
            "docs/README_AI_ENGINE.md": self.generate_ai_engine_readme(),
            # TODO: 其他 README 文件可以後續添加
        }
        
        for file_path, content in readmes.items():
            full_path = self.base_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 生成 README: {full_path}")
        
        print(f"🎉 完成！生成了 {len(readmes)} 個 README 文件")
        print(f"\n📍 生成位置: {self.base_dir.absolute()}")

if __name__ == "__main__":
    generator = CoreMultiLayerReadmeGenerator()
    generator.run_generation()
