# AIVA 隊列命名一致性檢查報告
==================================================

📊 **統計信息**
- 總隊列定義數: 9
- 符合標準: 9
- 需要修復: 0
- 合規率: 100.0%

✅ **符合標準的隊列**

- 📁 `services\features\function_authn_go\cmd\worker\main.go`
  - 隊列名: `findings.new` ✅
  - 語言: go
  - 類型: result

- 📁 `services\features\function_ssrf_go\cmd\worker\main.go`
  - 隊列名: `findings.new` ✅
  - 語言: go
  - 類型: result

- 📁 `services\features\function_ssrf_go\cmd\worker\main.go`
  - 隊列名: `tasks.function.ssrf` ✅
  - 語言: go
  - 類型: task

- 📁 `AIVA-git\services\features\function_sast_rust\src\worker.rs`
  - 隊列名: `tasks.function.sast` ✅
  - 語言: rust
  - 類型: task

- 📁 `AIVA-git\services\features\function_sast_rust\src\worker.rs`
  - 隊列名: `findings.new` ✅
  - 語言: rust
  - 類型: result

- 📁 `AIVA-git\services\scan\aiva_scan_node\src\index.ts`
  - 隊列名: `findings.new` ✅
  - 語言: typescript
  - 類型: result

- 📁 `AIVA-git\services\scan\aiva_scan_node\src\index.ts`
  - 隊列名: `task.scan.dynamic` ✅
  - 語言: typescript
  - 類型: task

- 📁 `AIVA-git\services\scan\info_gatherer_rust\src\main.rs`
  - 隊列名: `findings.new` ✅
  - 語言: rust
  - 類型: result

- 📁 `AIVA-git\services\scan\info_gatherer_rust\src\main.rs`
  - 隊列名: `tasks.scan.sensitive_info` ✅
  - 語言: rust
  - 類型: task
