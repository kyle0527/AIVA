# AIVA 修復計畫總覽

> 製作日期: 2026-03-11  
> 基準: branch `main` @ 987d677b + 本地未提交變更  
> 目標: 從「進階原型 35%」推進到「可執行首次完整掃描」

---

## 當前完成度

| 層面 | 完成度 | 狀態 |
|------|--------|------|
| 架構設計 | 85% | 設計完整，文件充足 |
| 核心框架 | 60% | FastAPI + 路由 + 神經網路骨架就位 |
| AI 決策引擎 | 40% | 模型存在但 decide() 介面未對接 |
| 功能模組 | 35% | XSS 完整，SQLi 殘缺，其餘骨架 |
| 端到端流程 | 20% | 13 步驟有設計但串不起來 |
| 部署能力 | 10% | Docker 全部過期 |
| 測試品質 | 5% | 6 個測試 / 505 個檔案 |
| **綜合** | **~35%** | |

---

## 修復計畫結構

修復計畫分為四份文件，按優先級排列：

| 文件 | 內容 | 預期效果 |
|------|------|----------|
| [01_PHASE1_CRITICAL_FIXES.md](01_PHASE1_CRITICAL_FIXES.md) | 關鍵阻塞修復（掃描流程串通） | 能走通第一次 POST /scan |
| [02_PHASE2_DEPLOYMENT_FIXES.md](02_PHASE2_DEPLOYMENT_FIXES.md) | 部署基礎建設修復 | docker build 能成功、DB 能初始化 |
| [03_PHASE3_FEATURE_COMPLETION.md](03_PHASE3_FEATURE_COMPLETION.md) | 功能模組補齊 + 品質提升 | 多漏洞類型支援 + 測試覆蓋 |
| [04_GIT_CLEANUP.md](04_GIT_CLEANUP.md) | Git 狀態清理 + 提交策略 | 乾淨的版本歷史 |

---

## 阻塞項摘要（按嚴重度排列）

### P0 — 紅色：程式無法正常啟動/掃描
1. `AttackCoordinator` import 斷裂 → 引用已刪的 `sqli_detector.py`
2. `CommanderCoordinator` 初始化因 #1 失敗 → `commander = None`
3. `/scan` 端點因 `commander is None` 無法走 Commander 模式
4. `RealDecisionEngine.decide()` 介面不匹配（期望 Tensor 收到 dict）

### P1 — 橘色：部署不可用
5. `Dockerfile.complete` 引用 6 個不存在的目錄
6. 入口點 `aiva_complete_launcher.py` 不存在
7. `entrypoint.sh` 不存在
8. Alembic 遷移已刪除，DB 無法自動初始化

### P2 — 黃色：功能不完整
9. SQLi 模組缺 `__main__.py` + detector 路徑斷裂
10. 12 個功能模組中只有 2 個有 `__main__.py`
11. 42 個 TODO/FIXME，29 個 NotImplementedError
12. 6 個測試檔 vs 505 個源碼檔

### P3 — 綠色：Git 清潔度
13. 80+ 個未提交變更（修改 + 刪除 + 新增混雜）
14. 部分刪除是有意重構，部分可能是誤刪

---

## 依賴關係圖

```
Phase 1 (P0 修復)
  ├─ 1.1 修復 SQLi import 路徑
  ├─ 1.2 修復 AttackCoordinator import
  ├─ 1.3 驗證 CommanderCoordinator 初始化
  ├─ 1.4 修復 decide() 介面
  └─ 1.5 端到端測試 POST /scan
          ↓
Phase 2 (部署修復)
  ├─ 2.1 修復 Dockerfile
  ├─ 2.2 建立入口點腳本
  ├─ 2.3 修復 DB 初始化
  └─ 2.4 docker build 測試
          ↓
Phase 3 (功能補齊)
  ├─ 3.1 補齊功能模組入口點
  ├─ 3.2 清理 NotImplementedError
  ├─ 3.3 補寫測試
  └─ 3.4 消除 TODO/FIXME
          ↓
Phase 4 (Git 清理)
  └─ 4.1 分批提交有意義的變更
```

---

## 程式碼規模參考

- `services/` 下 Python 檔案: **505 個**
- `services/` 下 Python 總行數: **172,991 行**
- 神經網路參數: **4,820,327 個** (17.35 MB 權重檔)
- FastAPI 路由: **11 個端點**
- 功能模組: **12 個** (XSS 完整、SQLi 半完成、其餘為骨架)
