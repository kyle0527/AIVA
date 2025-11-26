"""測試 StaticContentParser 是否能從 Juice Shop 首頁提取鏈接"""
import requests
from bs4 import BeautifulSoup

url = "http://localhost:3000"
r = requests.get(url)
soup = BeautifulSoup(r.text, "lxml")
links = soup.find_all("a")

print(f"✅ 找到 {len(links)} 個 <a> 標籤")
print("\n前 10 個鏈接:")
for i, a in enumerate(links[:10], 1):
    href = a.get("href")
    print(f"  [{i}] {href}")

print("\n\n📄 HTML 前 1000 字元:")
print(r.text[:1000])
