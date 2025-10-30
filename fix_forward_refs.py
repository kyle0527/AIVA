import re

with open("services/aiva_common/schemas/api_standards.py", "r", encoding="utf-8") as f:
    content = f.read()

# 修復前向引用
forward_refs = ["AsyncAPIOperationReply", "AsyncAPIMessageTrait", "AsyncAPITag", "OpenAPIExternalDocumentation", "OpenAPIReference"]
original_content = content

for ref_type in forward_refs:
    patterns = [
        (f"Union[{ref_type},", f"Union[\"{ref_type}\","),
        (f", {ref_type}]", f", \"{ref_type}\"]"),
        (f"Optional[{ref_type}]", f"Optional[\"{ref_type}\"]"),
        (f"List[{ref_type}]", f"List[\"{ref_type}\"]"),
        (f"Dict[str, {ref_type}]", f"Dict[str, \"{ref_type}\"]"),
    ]
    for pattern, replacement in patterns:
        content = content.replace(pattern, replacement)

if content != original_content:
    with open("services/aiva_common/schemas/api_standards.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("前向引用修復成功")
else:
    print("無需修復")

