#!/usr/bin/env python3
"""
AIVA Features 模組功能導向架構圖生成器
根據功能分類重新組織和生成架構圖表
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

class FunctionalArchitectureGenerator:
    """功能導向架構圖生成器"""
    
    def __init__(self, classification_file: str = "_out/architecture_diagrams/features_diagram_classification.json"):
        self.classification_file = Path(classification_file)
        self.output_dir = Path("_out/architecture_diagrams/functional")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入分類資料
        with open(self.classification_file, 'r', encoding='utf-8') as f:
            self.classification_data = json.load(f)
        
        # 功能分類配色
        self.category_colors = {
            "core": "#7c3aed",      # 紫色 - 核心功能
            "security": "#dc2626",   # 紅色 - 安全功能  
            "feature": "#2563eb",    # 藍色 - 業務功能
            "detail": "#059669",     # 綠色 - 支援功能
            "cross_lang": "#ea580c"  # 橙色 - 跨語言功能
        }
        
        # 語言配色
        self.language_colors = {
            "python": "#3776ab",
            "go": "#00ADD8", 
            "rust": "#CE422B"
        }
    
    def extract_functional_modules(self) -> Dict[str, Dict[str, List[str]]]:
        """提取功能模組結構"""
        functional_structure = defaultdict(lambda: defaultdict(list))
        
        classifications = self.classification_data["classifications"]
        
        for component_name, info in classifications.items():
            category = info["category"]
            language = info["language"]
            file_path = info["file_path"]
            
            # 根據檔案路徑推斷功能模組
            path_parts = file_path.replace("\\", "/").split("/")
            
            # 識別功能模組
            module_name = self._identify_module(path_parts, component_name)
            
            functional_structure[category][module_name].append({
                "name": component_name,
                "language": language,
                "file_path": file_path,
                "complexity": info["complexity"],
                "priority": info["priority"]
            })
        
        return functional_structure
    
    def _identify_module(self, path_parts: List[str], component_name: str) -> str:
        """識別功能模組名稱"""
        
        # SQL 注入相關
        if any("sqli" in part.lower() for part in path_parts) or "sqli" in component_name.lower():
            return "SQL_Injection_Detection"
            
        # XSS 相關
        if any("xss" in part.lower() for part in path_parts) or "xss" in component_name.lower():
            return "XSS_Detection"
            
        # SSRF 相關
        if any("ssrf" in part.lower() for part in path_parts) or "ssrf" in component_name.lower():
            return "SSRF_Detection"
            
        # 認證相關
        if any("authn" in part.lower() or "auth" in part.lower() for part in path_parts):
            return "Authentication_Security"
            
        # SAST 相關
        if any("sast" in part.lower() for part in path_parts) or "sast" in component_name.lower():
            return "Static_Analysis_SAST"
            
        # CSPM 相關
        if any("cspm" in part.lower() for part in path_parts) or "cspm" in component_name.lower():
            return "Cloud_Security_CSPM"
            
        # SCA 相關
        if any("sca" in part.lower() for part in path_parts) or "sca" in component_name.lower():
            return "Software_Composition_SCA"
            
        # 高價值目標
        if "high_value" in component_name.lower() or any("high" in part and "value" in part for part in path_parts):
            return "High_Value_Target"
            
        # 智能檢測
        if "smart" in component_name.lower() or "detection" in component_name.lower():
            return "Smart_Detection"
            
        # 功能管理
        if "manager" in component_name.lower() or "feature" in component_name.lower():
            return "Feature_Management"
            
        # 配置和模型
        if any(keyword in component_name.lower() for keyword in ["config", "model", "schema"]):
            return "Configuration_Models"
            
        # 默認按檔案名分組
        if len(path_parts) > 2:
            return path_parts[-1].replace('.py', '').replace('.go', '').replace('.rs', '')
        
        return "General_Utilities"
    
    def generate_core_functions_diagram(self, structure: Dict[str, Dict[str, List[str]]]) -> str:
        """生成核心功能架構圖"""
        
        core_modules = structure.get("core", {})
        
        diagram = """---
title: AIVA Features - 核心功能架構
config:
  theme: base
  themeVariables:
    primaryColor: "#7c3aed"
    primaryTextColor: "#fff"
    lineColor: "#6b7280"
---
flowchart TD
    CORE_HUB["🎯 核心功能控制中心<br/>61 個核心組件"]
    
"""
        
        # 核心模組
        for module_name, components in core_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            
            # 統計語言分佈
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            
            lang_info = ", ".join([f"{lang}: {count}" for lang, count in lang_stats.items()])
            
            diagram += f'    {safe_module}["{module_name}<br/>{component_count} 組件<br/>{lang_info}"]\n'
            diagram += f'    CORE_HUB --> {safe_module}\n'
        
        # 樣式定義
        diagram += f"""
    %% 核心功能樣式
    classDef coreStyle fill:{self.category_colors["core"]},stroke:#6b21a8,stroke-width:3px,color:#fff
    classDef hubStyle fill:#1f2937,stroke:#374151,stroke-width:4px,color:#fff
    
    class CORE_HUB hubStyle
"""
        
        # 應用樣式到所有核心模組
        core_module_names = [name.replace(" ", "_").replace("-", "_") for name in core_modules.keys()]
        if core_module_names:
            diagram += f"    class {','.join(core_module_names)} coreStyle\n"
        
        return diagram
    
    def generate_security_functions_diagram(self, structure: Dict[str, Dict[str, List[str]]]) -> str:
        """生成安全功能架構圖"""
        
        security_modules = structure.get("security", {})
        
        diagram = """---
title: AIVA Features - 安全功能架構
config:
  theme: base
  themeVariables:
    primaryColor: "#dc2626"
    primaryTextColor: "#fff"
    lineColor: "#6b7280"
---
flowchart TD
    SEC_HUB["🛡️ 安全功能控制中心<br/>2111 個安全組件<br/>78.4% 系統重心"]
    
    %% 主要安全模組群組
    SAST_GROUP["🔍 靜態分析群組"]
    VULN_GROUP["🚨 漏洞檢測群組"] 
    AUTH_GROUP["🔐 認證安全群組"]
    CLOUD_GROUP["☁️ 雲端安全群組"]
    
    SEC_HUB --> SAST_GROUP
    SEC_HUB --> VULN_GROUP
    SEC_HUB --> AUTH_GROUP
    SEC_HUB --> CLOUD_GROUP
    
"""
        
        # 分組安全模組
        sast_modules = {}
        vuln_modules = {}
        auth_modules = {}
        cloud_modules = {}
        other_modules = {}
        
        for module_name, components in security_modules.items():
            if "sast" in module_name.lower() or "static" in module_name.lower():
                sast_modules[module_name] = components
            elif any(keyword in module_name.lower() for keyword in ["sqli", "xss", "ssrf", "injection", "detection"]):
                vuln_modules[module_name] = components
            elif "auth" in module_name.lower():
                auth_modules[module_name] = components
            elif "cspm" in module_name.lower() or "cloud" in module_name.lower():
                cloud_modules[module_name] = components
            else:
                other_modules[module_name] = components
        
        # 生成各群組的子模組
        def add_modules_to_group(modules_dict, group_name):
            for module_name, components in modules_dict.items():
                safe_module = module_name.replace(" ", "_").replace("-", "_")
                component_count = len(components)
                
                # 語言統計
                lang_stats = defaultdict(int)
                for comp in components:
                    lang_stats[comp["language"]] += 1
                
                main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x])
                lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
                
                diagram_line = f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
                connection_line = f'    {group_name} --> {safe_module}\n'
                
                return diagram_line, connection_line, safe_module
        
        # 添加各群組模組
        module_styles = []
        
        for module_name, components in sast_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
            lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
            
            diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
            diagram += f'    SAST_GROUP --> {safe_module}\n'
            module_styles.append(safe_module)
        
        for module_name, components in vuln_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
            lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
            
            diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
            diagram += f'    VULN_GROUP --> {safe_module}\n'
            module_styles.append(safe_module)
        
        for module_name, components in auth_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
            lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
            
            diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
            diagram += f'    AUTH_GROUP --> {safe_module}\n'
            module_styles.append(safe_module)
        
        for module_name, components in cloud_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
            lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
            
            diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
            diagram += f'    CLOUD_GROUP --> {safe_module}\n'
            module_styles.append(safe_module)
        
        # 樣式定義
        diagram += f"""
    %% 安全功能樣式
    classDef securityStyle fill:{self.category_colors["security"]},stroke:#b91c1c,stroke-width:3px,color:#fff
    classDef hubStyle fill:#1f2937,stroke:#374151,stroke-width:4px,color:#fff
    classDef groupStyle fill:#7f1d1d,stroke:#991b1b,stroke-width:2px,color:#fff
    
    class SEC_HUB hubStyle
    class SAST_GROUP,VULN_GROUP,AUTH_GROUP,CLOUD_GROUP groupStyle
"""
        
        if module_styles:
            diagram += f"    class {','.join(module_styles)} securityStyle\n"
        
        return diagram
    
    def generate_business_features_diagram(self, structure: Dict[str, Dict[str, List[str]]]) -> str:
        """生成業務功能架構圖"""
        
        feature_modules = structure.get("feature", {})
        
        diagram = """---
title: AIVA Features - 業務功能架構
config:
  theme: base
  themeVariables:
    primaryColor: "#2563eb"
    primaryTextColor: "#fff"
    lineColor: "#6b7280"
---
flowchart TD
    BIZ_HUB["🏢 業務功能控制中心<br/>174 個業務組件<br/>6.5% 功能實現"]
    
"""
        
        module_styles = []
        for module_name, components in feature_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            
            # 語言統計
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            
            main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
            lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
            
            diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
            diagram += f'    BIZ_HUB --> {safe_module}\n'
            module_styles.append(safe_module)
        
        # 樣式定義
        diagram += f"""
    %% 業務功能樣式
    classDef featureStyle fill:{self.category_colors["feature"]},stroke:#1d4ed8,stroke-width:3px,color:#fff
    classDef hubStyle fill:#1f2937,stroke:#374151,stroke-width:4px,color:#fff
    
    class BIZ_HUB hubStyle
"""
        
        if module_styles:
            diagram += f"    class {','.join(module_styles)} featureStyle\n"
        
        return diagram
    
    def generate_support_functions_diagram(self, structure: Dict[str, Dict[str, List[str]]]) -> str:
        """生成支援功能架構圖"""
        
        detail_modules = structure.get("detail", {})
        
        diagram = """---
title: AIVA Features - 支援功能架構
config:
  theme: base
  themeVariables:
    primaryColor: "#059669"
    primaryTextColor: "#fff"
    lineColor: "#6b7280"
---
flowchart TD
    SUPPORT_HUB["🔧 支援功能控制中心<br/>346 個支援組件<br/>12.9% 基礎設施"]
    
"""
        
        module_styles = []
        for module_name, components in detail_modules.items():
            safe_module = module_name.replace(" ", "_").replace("-", "_")
            component_count = len(components)
            
            # 語言統計
            lang_stats = defaultdict(int)
            for comp in components:
                lang_stats[comp["language"]] += 1
            
            main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
            lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
            
            diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件<br/>主要: {main_lang}"]\n'
            diagram += f'    SUPPORT_HUB --> {safe_module}\n'
            module_styles.append(safe_module)
        
        # 樣式定義
        diagram += f"""
    %% 支援功能樣式
    classDef supportStyle fill:{self.category_colors["detail"]},stroke:#047857,stroke-width:3px,color:#fff
    classDef hubStyle fill:#1f2937,stroke:#374151,stroke-width:4px,color:#fff
    
    class SUPPORT_HUB hubStyle
"""
        
        if module_styles:
            diagram += f"    class {','.join(module_styles)} supportStyle\n"
        
        return diagram
    
    def generate_integrated_functional_diagram(self, structure: Dict[str, Dict[str, List[str]]]) -> str:
        """生成整合功能架構圖"""
        
        diagram = """---
title: AIVA Features - 整合功能架構總覽
config:
  theme: base
  themeVariables:
    primaryColor: "#1f2937"
    primaryTextColor: "#fff"
    lineColor: "#6b7280"
---
flowchart TD
    AIVA_FEATURES["🎯 AIVA Features 模組<br/>2692 個總組件<br/>多語言安全平台"]
    
    %% 四大功能分類
    CORE_LAYER["🔴 核心功能層<br/>61 組件 (2.3%)<br/>系統核心"]
    SECURITY_LAYER["🛡️ 安全功能層<br/>2111 組件 (78.4%)<br/>主要業務"]
    FEATURE_LAYER["🏢 業務功能層<br/>174 組件 (6.5%)<br/>功能實現"]
    SUPPORT_LAYER["🔧 支援功能層<br/>346 組件 (12.9%)<br/>基礎設施"]
    
    %% 主要連接
    AIVA_FEATURES --> CORE_LAYER
    AIVA_FEATURES --> SECURITY_LAYER  
    AIVA_FEATURES --> FEATURE_LAYER
    AIVA_FEATURES --> SUPPORT_LAYER
    
    %% 核心模組 (前5個重要模組)
"""
        
        # 添加各層的重要模組 (前3-5個)
        def add_top_modules(modules_dict, layer_name, max_count=3):
            # 按組件數量排序，取前幾個
            sorted_modules = sorted(modules_dict.items(), key=lambda x: len(x[1]), reverse=True)[:max_count]
            
            for module_name, components in sorted_modules:
                safe_module = f"{layer_name}_{module_name}".replace(" ", "_").replace("-", "_")
                component_count = len(components)
                
                # 語言統計
                lang_stats = defaultdict(int)
                for comp in components:
                    lang_stats[comp["language"]] += 1
                
                main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
                lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
                
                diagram_line = f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件"]\n'
                connection_line = f'    {layer_name} --> {safe_module}\n'
                
                return diagram_line, connection_line, safe_module
        
        # 添加核心模組
        core_modules = structure.get("core", {})
        core_styles = []
        if core_modules:
            sorted_core = sorted(core_modules.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            for module_name, components in sorted_core:
                safe_module = f"CORE_{module_name}".replace(" ", "_").replace("-", "_")
                component_count = len(components)
                lang_stats = defaultdict(int)
                for comp in components:
                    lang_stats[comp["language"]] += 1
                main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
                lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
                
                diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件"]\n'
                diagram += f'    CORE_LAYER --> {safe_module}\n'
                core_styles.append(safe_module)
        
        # 添加安全模組 (前5個)
        security_modules = structure.get("security", {})
        security_styles = []
        if security_modules:
            sorted_security = sorted(security_modules.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            for module_name, components in sorted_security:
                safe_module = f"SEC_{module_name}".replace(" ", "_").replace("-", "_")
                component_count = len(components)
                lang_stats = defaultdict(int)
                for comp in components:
                    lang_stats[comp["language"]] += 1
                main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
                lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
                
                diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件"]\n'
                diagram += f'    SECURITY_LAYER --> {safe_module}\n'
                security_styles.append(safe_module)
        
        # 添加業務模組 (前3個)
        feature_modules = structure.get("feature", {})
        feature_styles = []
        if feature_modules:
            sorted_feature = sorted(feature_modules.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            for module_name, components in sorted_feature:
                safe_module = f"FEAT_{module_name}".replace(" ", "_").replace("-", "_")
                component_count = len(components)
                lang_stats = defaultdict(int)
                for comp in components:
                    lang_stats[comp["language"]] += 1
                main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
                lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
                
                diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件"]\n'
                diagram += f'    FEATURE_LAYER --> {safe_module}\n'
                feature_styles.append(safe_module)
        
        # 添加支援模組 (前3個)
        detail_modules = structure.get("detail", {})
        support_styles = []
        if detail_modules:
            sorted_detail = sorted(detail_modules.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            for module_name, components in sorted_detail:
                safe_module = f"SUPP_{module_name}".replace(" ", "_").replace("-", "_")
                component_count = len(components)
                lang_stats = defaultdict(int)
                for comp in components:
                    lang_stats[comp["language"]] += 1
                main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
                lang_icon = {"python": "🐍", "go": "🐹", "rust": "🦀"}.get(main_lang, "📄")
                
                diagram += f'    {safe_module}["{lang_icon} {module_name}<br/>{component_count} 組件"]\n'
                diagram += f'    SUPPORT_LAYER --> {safe_module}\n'
                support_styles.append(safe_module)
        
        # 樣式定義
        diagram += f"""
    %% 整合架構樣式
    classDef aivaStyle fill:#1f2937,stroke:#374151,stroke-width:4px,color:#fff
    classDef coreStyle fill:{self.category_colors["core"]},stroke:#6b21a8,stroke-width:3px,color:#fff
    classDef securityStyle fill:{self.category_colors["security"]},stroke:#b91c1c,stroke-width:3px,color:#fff
    classDef featureStyle fill:{self.category_colors["feature"]},stroke:#1d4ed8,stroke-width:3px,color:#fff
    classDef supportStyle fill:{self.category_colors["detail"]},stroke:#047857,stroke-width:3px,color:#fff
    
    class AIVA_FEATURES aivaStyle
    class CORE_LAYER coreStyle
    class SECURITY_LAYER securityStyle
    class FEATURE_LAYER featureStyle
    class SUPPORT_LAYER supportStyle
"""
        
        # 應用模組樣式
        if core_styles:
            diagram += f"    class {','.join(core_styles)} coreStyle\n"
        if security_styles:
            diagram += f"    class {','.join(security_styles)} securityStyle\n"
        if feature_styles:
            diagram += f"    class {','.join(feature_styles)} featureStyle\n"
        if support_styles:
            diagram += f"    class {','.join(support_styles)} supportStyle\n"
        
        return diagram
    
    def run_functional_organization(self):
        """執行功能導向組織"""
        print("🎯 開始功能導向架構圖組織...")
        
        # 提取功能結構
        functional_structure = self.extract_functional_modules()
        
        # 統計信息
        print("📊 功能模組統計:")
        for category, modules in functional_structure.items():
            print(f"  {category}: {len(modules)} 個模組")
            for module_name, components in modules.items():
                print(f"    - {module_name}: {len(components)} 個組件")
        
        # 生成各種功能架構圖
        diagrams = {
            "FEATURES_CORE_FUNCTIONS.mmd": self.generate_core_functions_diagram(functional_structure),
            "FEATURES_SECURITY_FUNCTIONS.mmd": self.generate_security_functions_diagram(functional_structure), 
            "FEATURES_BUSINESS_FUNCTIONS.mmd": self.generate_business_features_diagram(functional_structure),
            "FEATURES_SUPPORT_FUNCTIONS.mmd": self.generate_support_functions_diagram(functional_structure),
            "FEATURES_INTEGRATED_FUNCTIONAL.mmd": self.generate_integrated_functional_diagram(functional_structure)
        }
        
        # 儲存圖表
        for filename, content in diagrams.items():
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 生成功能圖表: {output_path}")
        
        # 生成功能組織報告
        self.generate_functional_report(functional_structure)
        
        print(f"🎉 功能導向組織完成！生成 {len(diagrams)} 個架構圖")
        return functional_structure
    
    def generate_functional_report(self, structure: Dict[str, Dict[str, List[str]]]):
        """生成功能組織報告"""
        report_path = self.output_dir / "FUNCTIONAL_ORGANIZATION_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# AIVA Features 模組功能組織報告

## 📋 **組織概覽**

基於功能分類對 AIVA Features 模組的 2,692 個組件進行重新組織和架構設計。

### **功能分層架構**

```
🎯 AIVA Features 模組 (2692 組件)
├── 🔴 核心功能層 (61 組件, 2.3%)
├── 🛡️ 安全功能層 (2111 組件, 78.4%) ⭐ 主力
├── 🏢 業務功能層 (174 組件, 6.5%)
└── 🔧 支援功能層 (346 組件, 12.9%)
```

---

## 📊 **各層詳細分析**

""")
            
            # 分析各功能層
            for category, modules in structure.items():
                category_name = {
                    "core": "🔴 核心功能層",
                    "security": "🛡️ 安全功能層", 
                    "feature": "🏢 業務功能層",
                    "detail": "🔧 支援功能層"
                }.get(category, category)
                
                total_components = sum(len(comps) for comps in modules.values())
                
                f.write(f"""
### **{category_name}**

**總組件數**: {total_components} 個
**模組數**: {len(modules)} 個

| 模組名稱 | 組件數 | 主要語言 | 複雜度 |
|----------|-------|----------|--------|
""")
                
                # 按組件數量排序
                sorted_modules = sorted(modules.items(), key=lambda x: len(x[1]), reverse=True)
                
                for module_name, components in sorted_modules:
                    component_count = len(components)
                    
                    # 語言統計
                    lang_stats = defaultdict(int)
                    complexity_stats = defaultdict(int)
                    
                    for comp in components:
                        lang_stats[comp["language"]] += 1
                        complexity_stats[comp["complexity"]] += 1
                    
                    main_lang = max(lang_stats.keys(), key=lambda x: lang_stats[x]) if lang_stats else "unknown"
                    main_complexity = max(complexity_stats.keys(), key=lambda x: complexity_stats[x]) if complexity_stats else "unknown"
                    
                    f.write(f"| **{module_name}** | {component_count} | {main_lang} | {main_complexity} |\n")
            
            f.write(f"""

---

## 🎨 **生成的功能架構圖**

1. **核心功能架構**: `FEATURES_CORE_FUNCTIONS.mmd`
2. **安全功能架構**: `FEATURES_SECURITY_FUNCTIONS.mmd` 
3. **業務功能架構**: `FEATURES_BUSINESS_FUNCTIONS.mmd`
4. **支援功能架構**: `FEATURES_SUPPORT_FUNCTIONS.mmd`
5. **整合功能架構**: `FEATURES_INTEGRATED_FUNCTIONAL.mmd`

---

## 🔍 **功能組織洞察**

### **關鍵發現**

1. **安全功能絕對主導**: 78.4% 的組件集中在安全功能層
2. **Rust 是安全主力**: 安全功能主要由 Rust 實現
3. **Python 負責整合**: 核心功能和業務功能主要使用 Python
4. **Go 專注高效能**: 特定的高效能安全服務使用 Go

### **架構優勢**

- ✅ **功能分層清晰**: 四層架構職責分明
- ✅ **語言選擇合理**: 各語言發揮所長
- ✅ **安全重心突出**: 符合安全平台定位
- ✅ **模組化程度高**: 便於維護和擴展

---

**📝 報告版本**: v1.0  
**🔄 生成時間**: {self.classification_data['analysis_timestamp']}  
**👥 分析團隊**: AIVA Functional Architecture Team

""")
        
        print(f"📄 功能組織報告已生成: {report_path}")

if __name__ == "__main__":
    generator = FunctionalArchitectureGenerator()
    results = generator.run_functional_organization()
    
    print("\n🎯 功能組織完成！")
    print("📁 查看生成的功能架構圖: _out/architecture_diagrams/functional/")
    print("📊 查看功能組織報告: _out/architecture_diagrams/functional/FUNCTIONAL_ORGANIZATION_REPORT.md")