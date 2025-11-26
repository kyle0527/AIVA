"""分析 Rust Phase 0 輸出,確認是否能提供 Go 引擎需要的資訊"""

import json

# 讀取 Rust 輸出
with open('services/scan/engines/rust_engine/rust_output.json', encoding='utf-8') as f:
    content = f.read()
    json_start = content.index('{')
    data = json.loads(content[json_start:])

target = data['targets'][0]

print("=" * 60)
print("Rust Phase 0 掃描結果分析")
print("=" * 60)

# 1. 端點統計
endpoints = target.get('endpoints', [])
print(f"\n1. 發現的端點: {len(endpoints)} 個")
print(f"   - 有參數的端點: {len([e for e in endpoints if '?' in e.get('path', '')])}")
print(f"   - API 端點: {len([e for e in endpoints if '/api' in e.get('path', '')])}")
print(f"   - 高風險端點: {len([e for e in endpoints if e.get('risk_level') == 'high'])}")

# 2. JS Findings
js_findings = target.get('js_findings', [])
print(f"\n2. JS 分析結果: {len(js_findings)} 個發現")

# 按類型分組
finding_types = {}
for f in js_findings:
    ftype = f.get('finding_type', 'unknown')
    finding_types[ftype] = finding_types.get(ftype, 0) + 1

for ftype, count in sorted(finding_types.items()):
    print(f"   - {ftype}: {count}")

# 3. API 端點詳情
api_endpoints = [f for f in js_findings if f.get('finding_type') == 'api_endpoint']
print(f"\n3. API 端點詳情 (前10個):")
for i, ep in enumerate(api_endpoints[:10], 1):
    value = ep.get('value', '')
    print(f"   {i}. {value[:80]}")

# 4. 檢查是否有帶參數的 URL
print(f"\n4. 檢查是否有帶查詢參數的 URL:")
urls_with_params = [f for f in js_findings if '?' in f.get('value', '')]
print(f"   找到 {len(urls_with_params)} 個帶參數的 URL")
for url in urls_with_params[:5]:
    print(f"   - {url.get('value', '')[:80]}")

# 5. Go 引擎需求分析
print(f"\n5. Go 引擎需求分析:")
print(f"   ✅ Rust 提供了 {len(endpoints)} 個端點路徑")
print(f"   ✅ Rust 提供了 {len(api_endpoints)} 個 API 端點")

# 檢查是否有完整的帶參數 URL
complete_urls = [f.get('value') for f in js_findings if f.get('finding_type') == 'api_endpoint' and '://' in f.get('value', '')]
print(f"   ⚠️ 完整 URL: {len(complete_urls)} 個")

if complete_urls:
    print(f"\n6. 可直接用於 Go 引擎測試的完整 URL (前5個):")
    for url in complete_urls[:5]:
        print(f"   - {url}")
else:
    print(f"\n6. ❌ Rust 沒有提供完整的帶參數 URL")
    print(f"   Go 引擎需要類似: http://localhost:3000/api/users?id=1")
    print(f"   Rust 只提供了: /api/users (路徑)")

# 7. 結論
print(f"\n{'=' * 60}")
print("結論:")
print("=" * 60)
print("Rust Phase 0 提供的資訊:")
print("  ✅ 端點路徑列表 (如 /api/users)")
print("  ✅ 技術棧識別")
print("  ✅ JS 中發現的 API 端點")
print("  ❌ 沒有完整的帶參數 URL")
print("  ❌ 沒有參數名稱和類型")
print("\nGo 引擎的 SSRF 測試需要:")
print("  ✅ 基礎 URL: http://localhost:3000 (有)")
print("  ❌ 帶參數的完整端點: /api/fetch?url=xxx (缺)")
print("  ❌ 參數名稱: 如 url, redirect, callback (缺)")
print("\n建議:")
print("  1. Rust Phase 0 先爬取端點")
print("  2. Python 引擎爬取完整頁面,提取表單和參數")
print("  3. Go 引擎再對這些帶參數的端點測試 SSRF")
