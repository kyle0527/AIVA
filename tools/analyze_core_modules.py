#!/usr/bin/env python3
"""
AIVA 核心模組代碼分析工具
分析代碼複雜度、結構和品質指標
"""

import ast
from collections import defaultdict
import json
from pathlib import Path


def analyze_python_file(filepath):
    """分析單個 Python 檔案"""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # 統計基本指標
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

        # 計算複雜度指標
        async_functions = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
        decorators = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list]

        # 計算行數
        lines = content.split('\n')
        code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        comment_lines = [line for line in lines if line.strip().startswith('#')]

        # 分析導入依賴
        import_modules = []
        for imp in imports:
            if isinstance(imp, ast.Import):
                import_modules.extend([alias.name for alias in imp.names])
            elif isinstance(imp, ast.ImportFrom):
                module = imp.module or ''
                import_modules.append(module)

        # 計算函數平均長度
        function_lengths = []
        for func in functions + async_functions:
            if hasattr(func, 'lineno') and hasattr(func, 'end_lineno') and func.end_lineno is not None:
                length = func.end_lineno - func.lineno + 1
                function_lengths.append(length)

        avg_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 0

        return {
            'file': str(filepath.name),
            'total_lines': len(lines),
            'code_lines': len(code_lines),
            'comment_lines': len(comment_lines),
            'classes': len(classes),
            'functions': len(functions),
            'async_functions': len(async_functions),
            'imports': len(imports),
            'decorators': len(decorators),
            'avg_function_length': round(avg_function_length, 1),
            'max_function_length': max(function_lengths) if function_lengths else 0,
            'class_names': [cls.name for cls in classes],
            'function_names': [func.name for func in functions[:5]],
            'import_modules': list(set(import_modules)),
            'complexity_score': _calculate_complexity_score(classes, functions, avg_function_length, len(imports))
        }
    except Exception as e:
        return {'file': str(filepath), 'error': str(e)}

def _calculate_complexity_score(classes, functions, avg_func_len, imports):
    """計算複雜度分數 (0-100, 越高越複雜)"""
    score = 0
    score += len(classes) * 5  # 每個類別 +5 分
    score += len(functions) * 2  # 每個函數 +2 分
    score += max(0, (avg_func_len - 20) * 1)  # 超過20行的函數額外計分
    score += imports * 1  # 每個導入 +1 分
    return min(100, score)

def analyze_core_modules():
    """分析 AIVA 核心模組"""
    # 修正路徑為當前項目路徑
    core_path = Path('./services/core/aiva_core')
    results = []

    if not core_path.exists():
        print(f"[FAIL] 核心模組路徑不存在: {core_path.absolute()}")
        return results

    for py_file in core_path.rglob('*.py'):
        if '__pycache__' not in str(py_file) and '.backup' not in str(py_file):
            result = analyze_python_file(py_file)
            if 'error' not in result:
                results.append(result)
            else:
                print(f"[WARN]  分析文件失敗: {py_file} - {result['error']}")

    return results

def generate_analysis_report(results):
    """生成分析報告"""
    print('=' * 80)
    print('AIVA 核心模組代碼分析報告')
    print('=' * 80)
    print(f'總計分析檔案: {len(results)} 個')
    print()

    # 按代碼行數排序
    results_by_size = sorted(results, key=lambda x: x['code_lines'], reverse=True)

    print('[SEARCH] 按代碼規模排序 (前10個最大文件):')
    print('-' * 80)
    for i, result in enumerate(results_by_size[:10]):
        complexity = int(result.get("complexity_score", 0))
        print(f'{i+1:2d}. {result["file"]:45s} | 代碼: {result["code_lines"]:4d} 行 | 複雜度: {complexity:3d}')

    print('\n[BRAIN] AI 相關核心模組分析:')
    print('-' * 80)
    ai_files = [r for r in results if 'ai_' in r['file'] or 'bio_neuron' in r['file'] or 'nlg_' in r['file']]
    for result in sorted(ai_files, key=lambda x: x['code_lines'], reverse=True):
        print(f'[U+1F4C1] {result["file"]}')
        print(f'   代碼行數: {result["code_lines"]}, 類別: {result["classes"]}, 函數: {result["functions"]}')
        if result['class_names']:
            print(f'   主要類別: {", ".join(result["class_names"][:3])}')
        print()

    print('[FAST] 性能關鍵模組分析:')
    print('-' * 80)
    performance_files = [r for r in results if any(keyword in r['file'] for keyword in
                        ['optimized', 'parallel', 'execution', 'task_', 'cache'])]
    for result in sorted(performance_files, key=lambda x: x['code_lines'], reverse=True):
        print(f'[U+1F4C1] {result["file"]}')
        print(f'   代碼行數: {result["code_lines"]}, 異步函數: {result["async_functions"]}, 複雜度: {result.get("complexity_score", 0)}')
        print()

    # 複雜度分析
    print('[STATS] 複雜度統計:')
    print('-' * 80)
    complexity_scores = [r.get('complexity_score', 0) for r in results]
    high_complexity = [r for r in results if r.get('complexity_score', 0) > 50]

    if complexity_scores:
        print(f'平均複雜度: {sum(complexity_scores)/len(complexity_scores):.1f}')
    else:
        print('平均複雜度: 0.0')
    print(f'高複雜度文件 (>50): {len(high_complexity)} 個')

    if high_complexity:
        print('\n[ALERT] 需要重構的高複雜度文件:')
        for result in sorted(high_complexity, key=lambda x: x.get('complexity_score', 0), reverse=True):
            complexity = int(result.get("complexity_score", 0))
            max_func_len = int(result["max_function_length"])
            print(f'   {result["file"]:40s} | 複雜度: {complexity:3d} | 最長函數: {max_func_len:3d} 行')# 依賴分析
    print('\n[U+1F517] 依賴關係分析:')
    print('-' * 80)
    all_imports = defaultdict(int)
    for result in results:
        for module in result.get('import_modules', []):
            if module and not module.startswith('.'):
                all_imports[module] += 1

    common_imports = sorted(all_imports.items(), key=lambda x: x[1], reverse=True)[:10]
    print('最常用的外部依賴:')
    for module, count in common_imports:
        print(f'   {module:30s}: {count:2d} 次')

    return results

if __name__ == '__main__':
    # 修復 _calculate_complexity_score 函數
    def _calculate_complexity_score(classes, functions, avg_func_len, imports):
        """計算複雜度分數 (0-100, 越高越複雜)"""
        score = 0
        score += len(classes) * 5  # 每個類別 +5 分
        score += len(functions) * 2  # 每個函數 +2 分
        score += max(0, (avg_func_len - 20) * 1)  # 超過20行的函數額外計分
        score += imports * 1  # 每個導入 +1 分
        return min(100, score)

    # 重新定義 analyze_python_file 以包含修復的函數
    def analyze_python_file(filepath):
        """分析單個 Python 檔案"""
        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # 統計基本指標
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

            # 計算複雜度指標
            async_functions = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
            decorators = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list]

            # 計算行數
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            comment_lines = [line for line in lines if line.strip().startswith('#')]

            # 分析導入依賴
            import_modules = []
            for imp in imports:
                if isinstance(imp, ast.Import):
                    import_modules.extend([alias.name for alias in imp.names])
                elif isinstance(imp, ast.ImportFrom):
                    module = imp.module or ''
                    import_modules.append(module)

            # 計算函數平均長度
            function_lengths = []
            for func in functions + async_functions:
                if hasattr(func, 'lineno') and hasattr(func, 'end_lineno'):
                    length = (func.end_lineno or func.lineno) - func.lineno + 1
                    function_lengths.append(length)

            avg_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 0

            return {
                'file': str(filepath.name),
                'total_lines': len(lines),
                'code_lines': len(code_lines),
                'comment_lines': len(comment_lines),
                'classes': len(classes),
                'functions': len(functions),
                'async_functions': len(async_functions),
                'imports': len(imports),
                'decorators': len(decorators),
                'avg_function_length': round(avg_function_length, 1),
                'max_function_length': max(function_lengths) if function_lengths else 0,
                'class_names': [cls.name for cls in classes],
                'function_names': [func.name for func in functions[:5]],
                'import_modules': list(set(import_modules)),
                'complexity_score': _calculate_complexity_score(classes, functions, avg_function_length, len(imports))
            }
        except Exception as e:
            return {'file': str(filepath), 'error': str(e)}

    results = analyze_core_modules()
    generate_analysis_report(results)

    # 儲存詳細結果到 JSON
    with open('_out/core_module_analysis_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n[NOTE] 詳細分析結果已儲存到: _out/core_module_analysis_detailed.json')
