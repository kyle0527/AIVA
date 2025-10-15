# 🎯 AIVA 示例和演示

本目錄包含 AIVA 專案的各種示例程式和演示腳本。

## 📁 示例文件

### 🧠 AI 演示
- `demo_bio_neuron_agent.py` - BioNeuron AI 代理演示
- `demo_bio_neuron_master.py` - BioNeuron 主控制器演示

### 🔧 功能演示
- `demo_storage.py` - 儲存系統演示
- `demo_ui_panel.py` - UI 面板演示
- `demo_module_import_fix.py` - 模組導入修復演示

### 🚀 啟動腳本
- `start_ui_auto.py` - 自動啟動 UI 的 Python 版本
- `init_storage.py` - 初始化儲存系統

## 🎮 演示說明

### AI 代理演示
```bash
# BioNeuron AI 代理
python examples/demo_bio_neuron_agent.py

# 主控制器演示
python examples/demo_bio_neuron_master.py
```

### 存儲系統演示
```bash
# 初始化存儲
python examples/init_storage.py

# 存儲功能演示
python examples/demo_storage.py
```

### UI 相關演示
```bash
# UI 面板演示
python examples/demo_ui_panel.py

# 自動啟動 UI
python examples/start_ui_auto.py
```

## 🧠 AI 演示詳解

### BioNeuron Agent
展示 AIVA 的核心 AI 能力：
- 神經網路推理
- 強化學習
- 經驗積累
- 決策制定

```python
from examples.demo_bio_neuron_agent import BioNeuronDemo

# 創建演示實例
demo = BioNeuronDemo()

# 運行AI決策演示
result = demo.run_decision_demo()
print(f"AI 決策結果: {result}")
```

### Master Controller
展示主控制器的協調能力：
- 多模式切換
- 任務分派
- 結果整合
- 風險評估

```python
from examples.demo_bio_neuron_master import MasterDemo

# 創建主控演示
master = MasterDemo()

# 運行完整工作流程
await master.run_full_workflow()
```

## 💾 存儲演示

### 初始化存儲系統
```bash
# 設置資料庫
python examples/init_storage.py --setup-db

# 初始化向量數據庫
python examples/init_storage.py --setup-vector-db

# 載入初始知識庫
python examples/init_storage.py --load-knowledge
```

### 存儲功能演示
展示各種存儲功能：
- PostgreSQL 操作
- Redis 快取
- Neo4j 圖數據庫
- 向量數據庫

## 🎨 UI 演示

### UI 面板功能
- 即時狀態監控
- 交互式控制面板
- 結果視覺化
- 進度追蹤

### 自動化 UI 啟動
```bash
# 自動檢測並啟動最佳UI配置
python examples/start_ui_auto.py

# 指定配置啟動
python examples/start_ui_auto.py --config production
```

## 🔧 開發參考

### 自定義演示
基於現有演示創建自己的示例：

```python
# 參考模板
from examples.demo_bio_neuron_agent import BioNeuronDemo

class MyCustomDemo(BioNeuronDemo):
    def __init__(self):
        super().__init__()
        # 自定義初始化
    
    def my_custom_function(self):
        # 實現自定義功能
        pass

# 運行演示
if __name__ == "__main__":
    demo = MyCustomDemo()
    demo.run()
```

### 測試數據生成
使用演示腳本生成測試數據：

```bash
# 生成AI訓練數據
python examples/demo_bio_neuron_agent.py --generate-training-data

# 生成測試場景
python examples/demo_storage.py --create-test-scenarios
```

## 📊 演示報告

每個演示執行後會生成：
- 執行日誌
- 性能指標
- 結果截圖（UI演示）
- 數據導出

報告位置：`_out/demo_reports/`

## 🚀 快速體驗

### 5分鐘完整演示
```bash
# 1. 初始化系統
python examples/init_storage.py

# 2. 啟動AI演示
python examples/demo_bio_neuron_master.py --quick-demo

# 3. 查看UI演示
python examples/demo_ui_panel.py --auto-play
```

### 互動式演示
```bash
# 啟動互動式模式
python examples/demo_bio_neuron_agent.py --interactive

# 跟隨提示進行操作
# [1] 運行決策演示
# [2] 查看AI推理過程  
# [3] 測試強化學習
# [4] 退出
```

## 🛠️ 故障排除

### 常見問題
1. **模組導入錯誤**: 確保在專案根目錄執行
2. **資料庫連接失敗**: 檢查資料庫服務狀態
3. **AI模型未載入**: 運行初始化腳本

### 調試模式
```bash
# 啟用詳細日誌
python examples/demo_bio_neuron_agent.py --debug

# 步進式執行
python examples/demo_bio_neuron_agent.py --step-by-step
```

---

**用途**: 學習、演示、測試  
**維護者**: Development Team  
**最後更新**: 2025-10-16