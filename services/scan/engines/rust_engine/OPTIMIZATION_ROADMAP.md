# Rust Engine 優化路線圖

**原則**: 不破壞現有功能,漸進式優化  
**目標**: 從「可運作」→「高效運作」→「完整運作」

---

## 🎯 Phase A: 代碼品質優化 (不影響功能)

**預計時間**: 2-3 天  
**風險**: 低 (重構,不改邏輯)

### A1. 消除重複代碼 ⭐⭐⭐

**問題**:
```rust
// Fast 模式 (Lines 275-335)
for js_url in [main.js, runtime.js, vendor.js] {
    let js_content = fetch_page_content(&js_url).await;
    let findings = js_analyzer.analyze(&js_content, &js_url);
}

// Deep 模式 (Lines 337-407)
for js_file in [main.js, runtime.js, vendor.js, polyfills.js, ...] {
    let js_content = fetch_page_content(&js_url).await;  // 重複!
    let findings = js_analyzer.analyze(&js_content, &js_url);  // 重複!
}
```

**解決方案**:
```rust
async fn analyze_js_files(
    base_url: &str,
    js_files: &[&str],
    js_analyzer: &JsAnalyzer,
) -> Vec<JsFinding> {
    let mut all_findings = Vec::new();
    for js_file in js_files {
        let js_url = format!("{}/{}", base_url, js_file);
        if let Ok(content) = fetch_page_content(&js_url).await {
            all_findings.extend(js_analyzer.analyze(&content, &js_url));
        }
    }
    all_findings
}

// Fast 模式
let js_files = ["main.js", "runtime.js", "vendor.js"];
let findings = analyze_js_files(&base_url, &js_files, &js_analyzer).await;

// Deep 模式
let js_files = ["main.js", "runtime.js", "vendor.js", "polyfills.js", "scripts.js"];
let findings = analyze_js_files(&base_url, &js_files, &js_analyzer).await;
```

**收益**:
- 減少 ~60 行代碼
- 更易維護
- 不影響功能

---

### A2. 優化 Regex 編譯 ⭐⭐

**問題**:
```rust
// js_analyzer.rs 每次 analyze() 都重新編譯正則
pub fn analyze(&self, content: &str, file_path: &str) -> Vec<JsFinding> {
    let api_regex = Regex::new(r"/api/\w+").unwrap();  // 每次重新編譯!
    // ...
}
```

**解決方案**:
```rust
pub struct JsAnalyzer {
    api_regex: Regex,
    key_regex: Regex,
    domain_regex: Regex,
    // ... 在 new() 時編譯一次
}

impl JsAnalyzer {
    pub fn new() -> Self {
        Self {
            api_regex: Regex::new(r"/api/\w+").unwrap(),
            key_regex: Regex::new(r"sk_live_[a-zA-Z0-9]{24}").unwrap(),
            // ...
        }
    }
}
```

**收益**:
- 性能提升 ~15-20%
- 不改變功能

---

### A3. 添加 JS 下載錯誤處理 ⭐⭐⭐

**問題**:
```rust
let js_content = fetch_page_content(&js_url).await;  // 如果失敗會怎樣?
```

**解決方案**:
```rust
match fetch_page_content(&js_url).await {
    Ok(content) => {
        let findings = js_analyzer.analyze(&content, &js_url);
        all_js_findings.extend(findings);
        println!("  - {}: {} findings", js_file, findings.len());
    }
    Err(e) => {
        eprintln!("⚠️  無法下載 {}: {}", js_url, e);
        // 繼續處理其他文件,不中斷掃描
    }
}
```

**收益**:
- 更穩定
- 更好的用戶反饋
- 不影響成功案例

---

### A4. 添加 JS Finding 去重 ⭐

**問題**: 可能有重複的 findings (同一 API 端點出現多次)

**解決方案**:
```rust
use std::collections::HashSet;

fn deduplicate_findings(findings: Vec<JsFinding>) -> Vec<JsFinding> {
    let mut seen = HashSet::new();
    findings.into_iter()
        .filter(|f| {
            let key = format!("{}:{}:{}", f.finding_type, f.value, f.file_path);
            seen.insert(key)
        })
        .collect()
}
```

**收益**:
- 更乾淨的報告
- 不影響功能完整性

---

## 🔧 Phase B: 功能增強 (新功能,不影響現有)

**預計時間**: 3-5 天  
**風險**: 低-中 (新增,不修改現有)

### B1. 修復端點探測問題 ⭐⭐⭐⭐

**問題**: `EndpointDiscoverer.discover()` 實際探測回傳 0 結果

**調查步驟**:
1. 添加詳細日誌
```rust
println!("🔍 開始探測常見路徑...");
for path in common_paths {
    let test_url = format!("{}{}", base_url, path);
    println!("  測試: {}", test_url);
    match reqwest::get(&test_url).await {
        Ok(resp) => {
            println!("    ✅ {}", resp.status());
            if resp.status().is_success() {
                endpoints.push(...);
            }
        }
        Err(e) => println!("    ❌ {}", e),
    }
}
```

2. 檢查 `common_paths` 列表是否正確

3. 檢查 HTTP client 配置 (timeout, redirects)

**預期修復**:
- 實際探測應發現 10-20 個端點
- robots.txt, sitemap.xml 應正常解析

---

### B2. 增強技術棧檢測 ⭐⭐⭐

**當前問題**: 僅 3 個字符串檢查

**新實現**:
```rust
pub struct TechDetector {
    // 指紋庫
    frameworks: HashMap<String, Vec<Pattern>>,
    libraries: HashMap<String, Vec<Pattern>>,
}

struct Pattern {
    regex: Regex,
    confidence: f32,
}

impl TechDetector {
    pub fn new() -> Self {
        let mut frameworks = HashMap::new();
        
        // Angular
        frameworks.insert("Angular".to_string(), vec![
            Pattern { regex: Regex::new(r"@angular/core").unwrap(), confidence: 0.95 },
            Pattern { regex: Regex::new(r"ng-version").unwrap(), confidence: 0.90 },
            Pattern { regex: Regex::new(r"angular\.js").unwrap(), confidence: 0.85 },
        ]);
        
        // React
        frameworks.insert("React".to_string(), vec![
            Pattern { regex: Regex::new(r"react\.production\.min\.js").unwrap(), confidence: 0.95 },
            Pattern { regex: Regex::new(r"__REACT_DEVTOOLS").unwrap(), confidence: 0.90 },
        ]);
        
        // Express (從 headers)
        frameworks.insert("Express".to_string(), vec![
            Pattern { regex: Regex::new(r"X-Powered-By: Express").unwrap(), confidence: 0.98 },
        ]);
        
        // ...
        
        Self { frameworks, libraries }
    }
    
    pub fn detect(&self, html: &str, headers: &HeaderMap) -> Vec<Technology> {
        let mut detected = Vec::new();
        
        // 檢查 HTML 內容
        for (name, patterns) in &self.frameworks {
            let mut max_confidence = 0.0;
            for pattern in patterns {
                if pattern.regex.is_match(html) {
                    max_confidence = max_confidence.max(pattern.confidence);
                }
            }
            if max_confidence > 0.0 {
                detected.push(Technology {
                    name: name.clone(),
                    confidence: max_confidence,
                    evidence: "HTML content".to_string(),
                });
            }
        }
        
        // 檢查 HTTP headers
        // ...
        
        detected
    }
}
```

**收益**:
- 識別 30+ 種技術
- 提供信心評分
- 不影響現有簡單檢測

---

### B3. 添加更多 JS 文件來源 ⭐⭐

**當前**: 僅硬編碼 6 個檔名

**新實現**:
```rust
async fn discover_js_files(base_url: &str, html: &str) -> Vec<String> {
    let mut js_files = Vec::new();
    
    // 1. 從 HTML <script> 標籤提取
    let script_regex = Regex::new(r#"<script[^>]+src="([^"]+\.js)"#).unwrap();
    for cap in script_regex.captures_iter(html) {
        if let Some(src) = cap.get(1) {
            let js_url = resolve_url(base_url, src.as_str());
            js_files.push(js_url);
        }
    }
    
    // 2. 常見檔名 (fallback)
    for name in ["main.js", "runtime.js", "vendor.js", "polyfills.js"] {
        let url = format!("{}/{}", base_url, name);
        if !js_files.contains(&url) {
            js_files.push(url);
        }
    }
    
    js_files
}
```

**收益**:
- 自動發現所有 JS 文件
- 不遺漏任何來源
- 不影響現有硬編碼邏輯

---

### B4. JS 文件敏感資訊掃描 ⭐⭐

**問題**: 目前僅掃描 HTML,JS 文件內容未掃描敏感資訊

**解決方案**:
```rust
// 在 analyze_js_files() 中添加
for js_file in js_files {
    let content = fetch_page_content(&js_url).await?;
    
    // 現有: JS 分析
    let findings = js_analyzer.analyze(&content, &js_url);
    
    // 新增: 敏感資訊掃描
    let sensitive = scanner.scan(&content, &js_url);
    
    all_js_findings.extend(findings);
    all_sensitive_info.extend(sensitive);
}
```

**收益**:
- 檢測 JS 中的密碼、密鑰
- 更完整的掃描
- 不影響現有 JS 分析

---

## 🚀 Phase C: 性能優化 (提升效率)

**預計時間**: 2-3 天  
**風險**: 低 (性能改進,邏輯不變)

### C1. JS 文件下載快取 ⭐⭐

**問題**: 如果多個目標是同一網站,會重複下載相同 JS 文件

**解決方案**:
```rust
use std::sync::Arc;
use tokio::sync::RwLock;

struct JsFileCache {
    cache: Arc<RwLock<HashMap<String, String>>>,
}

impl JsFileCache {
    async fn get_or_fetch(&self, url: &str) -> Result<String, Error> {
        // 檢查快取
        {
            let cache = self.cache.read().await;
            if let Some(content) = cache.get(url) {
                return Ok(content.clone());
            }
        }
        
        // 下載
        let content = fetch_page_content(url).await?;
        
        // 寫入快取
        {
            let mut cache = self.cache.write().await;
            cache.insert(url.to_string(), content.clone());
        }
        
        Ok(content)
    }
}
```

**收益**:
- 掃描 4 個相同網站時節省 ~75% 下載時間
- 不影響單目標掃描

---

### C2. 並行 JS 文件下載 ⭐⭐⭐

**問題**: 目前串行下載 JS 文件

**解決方案**:
```rust
use futures::future::join_all;

async fn analyze_js_files_parallel(
    js_files: &[String],
    js_analyzer: &JsAnalyzer,
) -> Vec<JsFinding> {
    let futures: Vec<_> = js_files.iter()
        .map(|js_url| async move {
            match fetch_page_content(js_url).await {
                Ok(content) => js_analyzer.analyze(&content, js_url),
                Err(_) => Vec::new(),
            }
        })
        .collect();
    
    let results = join_all(futures).await;
    results.into_iter().flatten().collect()
}
```

**收益**:
- 單目標掃描時間減少 ~40-50%
- 不改變掃描結果

---

### C3. 端點探測速率限制 ⭐

**問題**: 如果常見路徑過多 (100+),可能觸發 WAF

**解決方案**:
```rust
use tokio::time::{sleep, Duration};

async fn probe_endpoints_with_rate_limit(
    paths: &[&str],
    base_url: &str,
    requests_per_second: u32,
) -> Vec<Endpoint> {
    let delay = Duration::from_millis(1000 / requests_per_second as u64);
    
    let mut endpoints = Vec::new();
    for path in paths {
        let url = format!("{}{}", base_url, path);
        
        if let Ok(resp) = reqwest::get(&url).await {
            if resp.status().is_success() {
                endpoints.push(Endpoint::from_response(resp, path));
            }
        }
        
        sleep(delay).await;  // 速率限制
    }
    
    endpoints
}
```

**收益**:
- 避免被 WAF 封鎖
- 可配置掃描速度

---

## 📋 優先級總結

### 立即執行 (本週)

1. **A3. 錯誤處理** - 2 小時,提升穩定性
2. **A1. 消除重複代碼** - 4 小時,提升可維護性
3. **A4. Finding 去重** - 1 小時,改善報告品質

### 下週執行

4. **B1. 修復端點探測** - 1-2 天,重要功能修復
5. **A2. Regex 優化** - 2 小時,性能提升
6. **B4. JS 敏感資訊掃描** - 4 小時,功能增強

### 後續執行

7. **B2. 增強技術棧檢測** - 2-3 天
8. **B3. 自動發現 JS 文件** - 1 天
9. **C2. 並行 JS 下載** - 4 小時
10. **C1. JS 文件快取** - 4 小時

---

## 📊 預期成果

完成所有優化後:

| 指標 | 當前 | 預期 | 提升 |
|------|------|------|------|
| **單目標掃描時間** | 0.8s | 0.4s | 50% |
| **4 目標掃描時間** | 2.0s | 1.2s | 40% |
| **端點發現數量** | 0 (實際探測) | 10-20 | ∞ |
| **技術棧識別** | 2-3 種 | 10-15 種 | 5x |
| **JS 文件覆蓋** | 6 個 | 15-30 個 | 5x |
| **代碼行數** | 2850 | 2950 | +100 (功能↑) |
| **重複代碼** | ~100 行 | ~20 行 | -80% |

---

## ⚠️ 風險控制

### 開發原則

1. **每個優化獨立 PR**
   - 小步快跑
   - 易於回滾

2. **優化前後測試**
   ```bash
   # 優化前
   ./test_before.sh > before.json
   
   # 優化
   # ...
   
   # 優化後
   ./test_after.sh > after.json
   
   # 對比
   diff before.json after.json  # 應該僅性能差異
   ```

3. **保留功能測試**
   ```rust
   #[cfg(test)]
   mod tests {
       #[tokio::test]
       async fn test_real_juice_shop_scan() {
           let result = scan("http://localhost:3000", "fast").await;
           assert!(result.js_findings.len() >= 80);  // 至少 80 個 findings
           assert!(result.technologies.len() >= 2);  // 至少 2 種技術
       }
   }
   ```

---

## 🎯 成功標準

所有優化完成後,必須滿足:

1. ✅ **功能不減**: 
   - Juice Shop 測試仍發現 80+ findings
   - API 端點仍正確提取
   - 技術棧仍正確識別

2. ✅ **性能提升**:
   - 單目標掃描 < 0.5 秒
   - 4 目標掃描 < 1.5 秒

3. ✅ **穩定性提升**:
   - 失敗 JS 下載不中斷掃描
   - 所有錯誤都有明確提示

4. ✅ **代碼品質**:
   - 重複代碼 < 50 行
   - 所有 TODO 清除
   - 所有 public 方法有文檔
