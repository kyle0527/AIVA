import httpx
from bs4 import BeautifulSoup

r = httpx.get('http://example.com')
soup = BeautifulSoup(r.text, 'lxml')
links = soup.find_all('a')
print(f'example.com 有 {len(links)} 個鏈接')
for a in links[:5]:
    print(f'  {a.get("href")}')
