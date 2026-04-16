import re
import os

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if ConfigDict already there and no class Config
    if 'class Config:' not in content:
        return

    # Add ConfigDict to pydantic import if not there
    if 'from pydantic import' in content and 'ConfigDict' not in content:
        content = re.sub(r'(from pydantic import .*?)(?=\n)', r'\1, ConfigDict', content)
    
    def replacer(match):
        block = match.group(0)
        args = []
        for line in block.split('\n')[1:]:
            line = line.strip()
            if not line or line.startswith('\"\"\"') or line.startswith('#'):
                continue
            if '=' in line:
                args.append(line)
                
        return '    model_config = ConfigDict(' + ', '.join(args) + ')\n'

    # Matches class config and its indented block
    pattern = re.compile(r'    class Config:\n(?:        .*?\n)*', re.MULTILINE)
    content = pattern.sub(replacer, content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Fixed {filepath}')

files = [
    r'c:\D\fold7\AIVA-git\services\aiva_common\schemas\analysis.py',
    r'c:\D\fold7\AIVA-git\services\aiva_common\schemas\decision.py',
    r'c:\D\fold7\AIVA-git\services\aiva_common\schemas\analysis\results.py',
    r'c:\D\fold7\AIVA-git\services\aiva_common\schemas\security_events.py',
    r'c:\D\fold7\AIVA-git\services\aiva_common\schemas\security\events.py',
    r'c:\D\fold7\AIVA-git\services\aiva_common\observability\__init__.py'
]

for f in files:
    if os.path.exists(f):
        fix_file(f)
