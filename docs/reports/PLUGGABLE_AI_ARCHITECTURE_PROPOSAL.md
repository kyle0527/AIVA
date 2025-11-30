# AIVA 可插拔 AI 系統架構建議

**參考來源**: Ray Serve, BentoML, FastAPI, TensorFlow Serving 最佳實踐

---

## 🎯 核心問題

您提出的關鍵問題:
1. **AI 不是組件** - 目前是硬編碼,無法插拔
2. **不像程序運行** - 需要執行腳本或命令,而非作為服務運行
3. **權重未接上** - 5M 參數的 BioNeuron 沒有實際載入

---

## 📐 業界標準架構

### 1. **Model Serving 層 (Ray Serve/BentoML 模式)**

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  (HTTP/gRPC Endpoints)                  │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│      Model Registry & Loader            │
│  - 插件發現 (Plugin Discovery)           │
│  - 權重管理 (Weight Manager)             │
│  - 版本控制 (Version Control)            │
└─────────────┬───────────────────────────┘
              │
      ┌───────┴────────┐
      │                │
┌─────▼─────┐   ┌─────▼─────┐
│ AI Model  │   │ AI Model  │
│ Plugin #1 │   │ Plugin #2 │
│ (BioNeuron)│   │ (RAG)    │
└───────────┘   └───────────┘
```

### 2. **FastAPI Lifespan 模式 (啟動/關閉管理)**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 載入所有 AI 模型
    print("🚀 Loading AI models...")
    app.state.models = await load_all_models()
    yield
    # Shutdown: 釋放資源
    print("🛑 Unloading models...")
    await unload_all_models(app.state.models)

app = FastAPI(lifespan=lifespan)
```

---

## 🏗️ AIVA 具體實現方案

### **架構 1: 插件式 AI 引擎**

#### 1.1 基礎接口 (Interface)

```python
# services/aiva_common/ai/plugin_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path

class AIModelPlugin(ABC):
    """AI 模型插件基類"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模型名稱"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """模型版本"""
        pass
    
    @abstractmethod
    async def load_weights(self, weights_path: Path) -> None:
        """載入權重"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """推理"""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """卸載模型"""
        pass
```

#### 1.2 BioNeuron 插件實現

```python
# services/core/aiva_core/cognitive_core/neural/bio_neuron_plugin.py
import torch
import logging
from pathlib import Path
from typing import Any, Dict

from services.aiva_common.ai.plugin_interface import AIModelPlugin
from .real_neural_core import RealScalableBioNet

logger = logging.getLogger(__name__)

class BioNeuronPlugin(AIModelPlugin):
    """BioNeuron 5M 參數模型插件"""
    
    def __init__(self):
        self._model: RealScalableBioNet | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @property
    def name(self) -> str:
        return "BioNeuron"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def load_weights(self, weights_path: Path) -> None:
        """載入 5M 參數權重"""
        logger.info(f"Loading BioNeuron weights from {weights_path}")
        
        # 初始化模型
        self._model = RealScalableBioNet(
            input_size=512,
            hidden_sizes=[1024, 2048, 1024],
            output_size=256
        )
        
        # 載入權重
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self._device)
            self._model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded {self._model.get_total_parameters():,} parameters")
        else:
            logger.warning(f"⚠️ Weight file not found: {weights_path}")
            logger.info("Using randomly initialized weights")
        
        self._model.to(self._device)
        self._model.eval()
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """執行推理"""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")
        
        # 將輸入轉為 tensor
        input_tensor = torch.tensor(input_data["features"]).to(self._device)
        
        # 推理
        with torch.no_grad():
            output = self._model(input_tensor)
        
        return {
            "predictions": output.cpu().numpy().tolist(),
            "model": self.name,
            "version": self.version
        }
    
    async def unload(self) -> None:
        """卸載模型並釋放 GPU 記憶體"""
        if self._model is not None:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"✅ Unloaded {self.name}")
```

#### 1.3 模型註冊表

```python
# services/aiva_common/ai/model_registry.py
import logging
from pathlib import Path
from typing import Dict

from .plugin_interface import AIModelPlugin

logger = logging.getLogger(__name__)

class ModelRegistry:
    """AI 模型註冊表 - 管理所有可插拔模型"""
    
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        self._models: Dict[str, AIModelPlugin] = {}
    
    def register(self, plugin: AIModelPlugin) -> None:
        """註冊新模型"""
        self._models[plugin.name] = plugin
        logger.info(f"Registered model: {plugin.name} v{plugin.version}")
    
    async def load_all(self) -> None:
        """載入所有已註冊的模型"""
        logger.info(f"Loading {len(self._models)} models...")
        
        for name, plugin in self._models.items():
            weights_path = self.weights_dir / f"{name.lower()}_weights.pt"
            try:
                await plugin.load_weights(weights_path)
                logger.info(f"✅ {name} loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load {name}: {e}")
    
    async def unload_all(self) -> None:
        """卸載所有模型"""
        for name, plugin in self._models.items():
            await plugin.unload()
    
    def get(self, model_name: str) -> AIModelPlugin:
        """獲取模型插件"""
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found")
        return self._models[model_name]
    
    @property
    def available_models(self) -> list[str]:
        """返回所有可用模型名稱"""
        return list(self._models.keys())
```

#### 1.4 FastAPI 整合

```python
# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pathlib import Path
import logging

from services.aiva_common.ai.model_registry import ModelRegistry
from services.core.aiva_core.cognitive_core.neural.bio_neuron_plugin import BioNeuronPlugin
# 未來可以添加更多插件
# from services.core.aiva_core.cognitive_core.rag.rag_plugin import RAGPlugin

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理 - 在啟動時載入模型,關閉時釋放"""
    
    # Startup
    logger.info("🚀 AIVA Starting up...")
    
    # 初始化模型註冊表
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    registry = ModelRegistry(weights_dir)
    
    # 註冊所有 AI 模型插件
    registry.register(BioNeuronPlugin())
    # registry.register(RAGPlugin())  # 未來添加
    
    # 載入模型權重
    await registry.load_all()
    
    # 將註冊表存入 app state
    app.state.model_registry = registry
    
    logger.info(f"✅ Loaded models: {registry.available_models}")
    logger.info("🎉 AIVA ready to serve")
    
    yield  # 應用運行期間
    
    # Shutdown
    logger.info("🛑 AIVA shutting down...")
    await registry.unload_all()
    logger.info("✅ All models unloaded")

# 創建 FastAPI 應用
app = FastAPI(
    title="AIVA AI Platform",
    version="2.0.0",
    lifespan=lifespan  # 關鍵: 使用 lifespan 管理模型
)

@app.get("/models")
async def list_models():
    """列出所有可用模型"""
    return {
        "models": app.state.model_registry.available_models
    }

@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: dict):
    """使用指定模型進行推理"""
    try:
        model = app.state.model_registry.get(model_name)
        result = await model.predict(input_data)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 🎬 啟動方式

### **作為服務運行 (像程序一樣)**

```bash
# 生產環境
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# 開發環境
uvicorn api.main:app --reload
```

### **Docker 容器化**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install -r requirements.txt

# 複製代碼
COPY . .

# 下載/掛載權重文件
VOLUME /app/weights

# 暴露端口
EXPOSE 8000

# 啟動服務 (作為守護進程)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 💾 權重管理

### **方案 1: 本地文件**

```
weights/
├── bioneuron_weights.pt      # 5M 參數
├── rag_embeddings.pt         # RAG 向量
└── model_metadata.json       # 模型元數據
```

### **方案 2: 雲端存儲 (S3/Minio)**

```python
# services/aiva_common/ai/weight_loader.py
import boto3
from pathlib import Path

class WeightLoader:
    def __init__(self, s3_bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
    
    async def download_weights(self, model_name: str, local_path: Path):
        """從 S3 下載權重"""
        s3_key = f"models/{model_name}/weights.pt"
        self.s3.download_file(self.bucket, s3_key, str(local_path))
```

---

## 🔄 對比總結

| 維度 | ❌ 目前狀況 | ✅ 改進後 |
|------|------------|----------|
| **AI 組件** | 硬編碼在各處 | 插件式,可動態註冊 |
| **啟動方式** | 執行腳本/命令 | FastAPI 服務 (uvicorn) |
| **權重載入** | 未實現 | 啟動時自動載入 |
| **資源管理** | 手動管理 | Lifespan 自動管理 |
| **部署方式** | 複雜腳本 | Docker 容器化 |
| **可擴展性** | 難以添加新模型 | 實現接口即可插入 |

---

## 🚀 實施步驟

1. **Phase 1: 基礎架構** (3 天)
   - 實現 `AIModelPlugin` 接口
   - 創建 `ModelRegistry`
   - 修改 FastAPI `main.py` 添加 lifespan

2. **Phase 2: BioNeuron 插件** (2 天)
   - 實現 `BioNeuronPlugin`
   - 準備/訓練權重文件
   - 測試載入和推理

3. **Phase 3: 其他模型遷移** (5 天)
   - RAG Engine → `RAGPlugin`
   - Decision Agent → `DecisionPlugin`
   - Attack Executor → `AttackPlugin`

4. **Phase 4: 容器化部署** (2 天)
   - 編寫 Dockerfile
   - 配置 docker-compose
   - CI/CD 整合

---

## 📚 參考資源

- **Ray Serve**: https://docs.ray.io/en/latest/serve/
- **BentoML**: https://github.com/bentoml/BentoML
- **FastAPI Lifespan**: https://fastapi.tiangolo.com/advanced/events/
- **TorchServe**: https://pytorch.org/serve/

這套方案將 AIVA 轉變為真正的生產級 AI 服務平台! 🎯
