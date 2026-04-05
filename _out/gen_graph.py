import os
import ast
from collections import defaultdict

CORE_DIR = r'C:\D\fold7\AIVA-git\services\core\aiva_core'
CORE_PREFIX = 'services.core.aiva_core.'

files_ast = {}
dependencies = defaultdict(list)
reverse_deps = defaultdict(list)

# 1. Collect all py files and map them to their absolute theoretical import strings
def filepath_to_module(path):
    rel = os.path.relpath(path, CORE_DIR)
    if rel == '.': return ''
    mod = rel.replace('\\', '.').replace('/', '.')
    if mod.endswith('.py'): mod = mod[:-3]
    if mod.endswith('.__init__'): mod = mod[:-9]
    return mod

module_to_filepath = {}
for root, dirs, files in os.walk(CORE_DIR):
    if '__pycache__' in root or '_archive' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, CORE_DIR).replace(os.sep, '/')
            mod_name = filepath_to_module(full_path)
            module_to_filepath[mod_name] = rel_path
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
                try:
                    tree = ast.parse(file.read(), filename=full_path)
                    files_ast[rel_path] = (mod_name, tree, rel_path, full_path)
                except Exception:
                    pass

# 2. Parse imports
for rel_path, (mod_name, tree, _, full_path) in files_ast.items():
    current_pkg = mod_name.rsplit('.', 1)[0] if '.' in mod_name else ''
    if mod_name.endswith('__init__'):
        current_pkg = mod_name[:-9]
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imp = alias.name
                if imp.startswith(CORE_PREFIX):
                    target_mod = imp[len(CORE_PREFIX):]
                    dependencies[rel_path].append(target_mod)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                level = node.level
                imp = node.module
                
                if level > 0:
                    # relative import
                    parts = current_pkg.split('.') if current_pkg else []
                    if level > 1:
                        parts = parts[:-(level-1)]
                    base = '.'.join(parts)
                    target_mod = base + '.' + imp if base else imp
                    dependencies[rel_path].append(target_mod)
                elif imp.startswith(CORE_PREFIX):
                    target_mod = imp[len(CORE_PREFIX):]
                    dependencies[rel_path].append(target_mod)
            else:
                level = node.level
                if level > 0:
                    parts = current_pkg.split('.') if current_pkg else []
                    if level > 1:
                        parts = parts[:-(level-1)]
                    target_mod = '.'.join(parts)
                    dependencies[rel_path].append(target_mod)

# Normalize and map back to files
file_deps = defaultdict(set)
file_rev_deps = defaultdict(set)

for rel_path, deps in dependencies.items():
    for d in deps:
        # Best effort matching modulo name to filepath
        for m, fpath in module_to_filepath.items():
            if d == m or d.startswith(m + '.') or (m and d == m):
                if fpath != rel_path:
                    file_deps[rel_path].add(fpath)
                    file_rev_deps[fpath].add(rel_path)

# Generate Markdown
md = ['# AIVA Core 檔案相依性與關聯映射矩陣\n']
md.append('本文件透過靜態程式碼分析 (AST Parsing) 抽取出 AIVA 各個檔案之間真實的 Import 引用與相依關係。\n')
md.append('> **閱讀指南**：\n> - 🔗 **呼叫了誰 (Depends On)**：代表本檔案「依賴」並使用了哪些檔案的功能。\n> - 🎯 **被誰呼叫 (Used By)**：代表哪些檔案「依賴」並需要本檔案。\n\n')

sorted_files = sorted(files_ast.keys())
current_folder = ''
for f in sorted_files:
    folder = f.split('/')[0] if '/' in f else 'Root'
    if folder != current_folder:
        md.append(f'## 📂 {folder.upper()} 模組\n')
        current_folder = folder
    
    md.append(f'### 📄 `{f}`\n')
    
    deps = sorted(list(file_deps[f]))
    if deps:
        md.append('**🔗 呼叫了誰 (Depends On):**\n')
        for d in deps: md.append(f'- `{d}`\n')
    else:
        md.append('**🔗 呼叫了誰:** (無內部依賴/底層)\n')
        
    rev_deps = sorted(list(file_rev_deps[f]))
    if rev_deps:
        md.append('\n**🎯 被誰呼叫 (Used By):**\n')
        for d in rev_deps: md.append(f'- `{d}`\n')
    else:
        md.append('\n**🎯 被誰呼叫:** (頂層腳本或未被明確引用的動態模組)\n')
        
    md.append('\n---\n')

out_path = r'C:\Users\User\.gemini\antigravity\brain\9b3d8e97-cf75-4e37-92a0-58055e2037fd\artifacts\aiva_core_file_relationships.md'
with open(out_path, 'w', encoding='utf-8') as mf:
    mf.write(''.join(md))
print('Graph generation completed!')
