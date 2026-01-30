# AIVA Core 分層依賴配置指南

## ⚠️ 當前環境狀態

**✅ 全局環境已安裝完整依賴集（2026-01-09 驗證）**

當前全局 Python 環境已包含所有 full.txt 中的依賴，可直接使用。  
以下配置文件供新環境部署或虛擬環境隔離使用。

## 📌 檢查當前環境

```powershell
# 快速檢查關鍵依賴
python -c "import torch, transformers, fastapi, pydantic; print('✅ 完整環境已就緒')"

# 查看完整依賴列表
pip list | findstr /I "torch transformers fastapi pydantic"
```

---

本目錄包含不同場景的依賴配置文件，支持按需安裝。

## 📦 依賴層級

```
minimal.txt (65 MB, <1s)
    ↓
  web.txt (95 MB, ~2s)
    ↓
   ai.txt (2.6 GB, ~10s)
    ↓
  full.txt (4.5 GB, ~15s)
    ↓
   dev.txt (4.7 GB, ~15s)
```

## 🎯 使用場景

### 1. CLI 驗證 / 參數檢查
```bash
pip install -r requirements/minimal.txt
```
- **大小**: 65 MB
- **啟動**: <1秒
- **包含**: pydantic, loguru, numpy
- **用途**: 參數驗證、基礎檢查、單元測試

### 2. Web API 服務（無AI）
```bash
pip install -r requirements/web.txt
```
- **大小**: 95 MB
- **啟動**: ~2秒
- **包含**: minimal + fastapi, uvicorn, requests
- **用途**: REST API、工具調用、遠程控制

### 3. AI 能力服務
```bash
pip install -r requirements/ai.txt
```
- **大小**: 2.6 GB
- **啟動**: ~10秒
- **包含**: web + torch, sentence-transformers
- **用途**: AI決策、強化學習、RAG系統

### 4. 完整功能
```bash
pip install -r requirements/full.txt
```
- **大小**: 4.5 GB
- **啟動**: ~15秒
- **包含**: ai + transformers, spacy, scikit-learn
- **用途**: 所有功能、生產環境

### 5. 開發環境
```bash
pip install -r requirements/dev.txt
```
- **大小**: 4.7 GB
- **啟動**: ~15秒
- **包含**: full + 測試工具、代碼檢查、文檔工具
- **用途**: 本地開發、調試、測試

## 🔧 環境變數控制

設置 `AIVA_MODE` 環境變數來控制功能級別：

```bash
# 最小模式（僅基礎功能）
export AIVA_MODE=minimal

# Web 模式（Web API）
export AIVA_MODE=web

# AI 模式（AI 能力）
export AIVA_MODE=ai

# 完整模式（默認）
export AIVA_MODE=full
```

## 📊 性能對比

| 配置 | 磁盤 | 內存 | 啟動 | AI | Web | 適用場景 |
|------|------|------|------|----|----|----------|
| minimal | 65MB | 50MB | <1s | ❌ | ❌ | CI/CD、測試 |
| web | 95MB | 100MB | ~2s | ❌ | ✅ | API服務 |
| ai | 2.6GB | 1GB | ~10s | ✅ | ✅ | AI服務 |
| full | 4.5GB | 2GB | ~15s | ✅ | ✅ | 生產完整 |
| dev | 4.7GB | 2GB | ~15s | ✅ | ✅ | 開發環境 |

## 🚀 遷移指南

### 從完整安裝遷移到分層安裝

```bash
# 1. 卸載現有依賴
pip uninstall -r requirements.txt -y

# 2. 根據需求安裝對應層級
pip install -r requirements/ai.txt

# 3. 驗證功能
python -c "from services.core.aiva_core import __version__; print(__version__)"
```

### Docker 多階段構建

```dockerfile
# 階段 1: 基礎鏡像（minimal）
FROM python:3.10-slim as base
COPY requirements/minimal.txt .
RUN pip install -r minimal.txt

# 階段 2: Web 服務鏡像
FROM base as web
COPY requirements/web.txt .
RUN pip install -r web.txt

# 階段 3: AI 服務鏡像
FROM web as ai
COPY requirements/ai.txt .
RUN pip install -r ai.txt

# 最終鏡像根據需求選擇
FROM ai as final
```

## 💡 最佳實踐

### CI/CD 環境
```yaml
# .gitlab-ci.yml
test:
  script:
    - pip install -r requirements/minimal.txt
    - pytest tests/
  # 節省 98% 構建時間
```

### 開發環境
```bash
# 首次安裝
pip install -r requirements/dev.txt

# 日常使用
export AIVA_MODE=full
python -m services.core.aiva_core
```

### 生產環境
```bash
# API 服務器（無AI）
pip install -r requirements/web.txt
export AIVA_MODE=web

# AI 服務器
pip install -r requirements/ai.txt
export AIVA_MODE=ai
```

## 📝 維護建議

### 添加新依賴時
1. 確定依賴屬於哪個層級
2. 添加到對應的 .txt 文件
3. 更新此 README
4. 測試各層級安裝

### 升級依賴時
```bash
# 升級特定層級
pip install -r requirements/ai.txt --upgrade

# 生成新的鎖定文件
pip freeze > requirements/ai.lock
```

## 🔗 相關文檔

- [依賴完整分析](../DEPENDENCY_ANALYSIS.md)
- [依賴優化指南](../DEPENDENCY_OPTIMIZATION.md)
- [主 requirements.txt](../requirements.txt)

---

**維護者**: AIVA Team  
**最後更新**: 2026-01-09
