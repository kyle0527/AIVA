from pathlib import Path
import re

root_dir = Path(__file__).parent.parent.parent.parent
p = root_dir / "DATA_CONTRACT_UPDATE.md"
s = p.read_text(encoding="utf-8")
lines = s.splitlines()
issues = []

# Check ordered list prefixes not equal to '1.'
for i, line in enumerate(lines, start=1):
    m = re.match(r"^(\s*)(\d+)\.\s+", line)
    if m:
        num = int(m.group(2))
        if num != 1:
            issues.append((i, 'ol-prefix', line.strip()))

# Check headings increment by more than 1 level
last_h = 0
for i, line in enumerate(lines, start=1):
    m = re.match(r"^(#+)\s+", line)
    if m:
        level = len(m.group(1))
        if last_h and level - last_h > 1:
            issues.append((i, 'heading-increment', line.strip()))
        last_h = level

# Check fenced code blocks surrounded by blank lines
in_fence = False
for i, line in enumerate(lines, start=1):
    if re.match(r"^```", line):
        if not in_fence:
            in_fence = True
            # check previous line
            if i>1 and lines[i-2].strip() != "":
                issues.append((i, 'blanks-around-fences-above', lines[i-2].strip()))
        else:
            in_fence = False
            # check next line
            if i < len(lines) and lines[i].strip() != "":
                issues.append((i, 'blanks-around-fences-below', lines[i].strip()))

out_path = root_dir / "_out" / "markdown_check_out.txt"
out_path.parent.mkdir(exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    f.write(f'found {len(issues)} issues\n')
    for it in issues:
        # write a safe representation to avoid terminal encoding issues
        line_no, kind, text = it
        f.write(f"{line_no}\t{kind}\t{text}\n")
print(f"wrote results to {out_path}")
