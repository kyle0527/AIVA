import re

text = "掃描 http://localhost:3000"

patterns = [
    r"(幫我|幫忙|請|麻煩).*(掃描|scan|測試|test|攻擊|attack)\s*(?P<target>https?://\S+)",
    r"^(掃描|scan|測試|test)\s+(?P<url>https?://\S+)",
    r"(掃描|scan|測試|test|攻擊|attack)\s+(?P<target_url>https?://\S+)",
]

print(f"測試文本: '{text}'")
print()

for i, pattern in enumerate(patterns):
    match = re.search(pattern, text)
    if match:
        print(f"Pattern {i}: ✅ 匹配成功")
        print(f"  Groups: {match.groupdict()}")
    else:
        print(f"Pattern {i}: ❌ 無匹配")
