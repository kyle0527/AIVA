#!/usr/bin/env python3
"""
AIVA 跨語言警告分析工具
用於詳細分析和分類跨語言驗證中的警告
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Any

class CrossLanguageWarningAnalyzer:
    """跨語言警告分析器"""
    
    def __init__(self, report_file: str = "cross_language_validation_report.json"):
        self.report_file = Path(report_file)
        self.warnings = []
        self.analysis = {}
        
    def load_validation_report(self) -> bool:
        """載入驗證報告"""
        try:
            if not self.report_file.exists():
                print(f"❌ 找不到驗證報告文件: {self.report_file}")
                return False
                
            with open(self.report_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.warnings = [i for i in data.get('issues', []) if i.get('severity') == 'warning']
            print(f"✅ 載入 {len(self.warnings)} 個警告")
            return True
            
        except Exception as e:
            print(f"❌ 載入驗證報告失敗: {e}")
            return False
    
    def analyze_warnings(self) -> Dict[str, Any]:
        """詳細分析警告"""
        analysis = {
            'analysis_date': datetime.now().isoformat(),
            'total_warnings': len(self.warnings),
            'by_category': defaultdict(list),
            'by_schema': defaultdict(list), 
            'by_language': defaultdict(list),
            'by_message_pattern': defaultdict(list),
            'specific_issues': defaultdict(list)
        }
        
        # 分類每個警告
        for warning in self.warnings:
            category = warning.get('category', 'unknown')
            schema = warning.get('schema_name', 'unknown')
            languages = warning.get('languages', [])
            message = warning.get('message', '')
            field_name = warning.get('field_name', '')
            
            # 基本分類
            analysis['by_category'][category].append(warning)
            analysis['by_schema'][schema].append(warning)
            
            for lang in languages:
                analysis['by_language'][lang].append(warning)
            
            # 按消息模式詳細分類
            if '類型' in message and '沒有映射' in message:
                pattern = '類型映射缺失'
                # 提取具體缺失的類型
                if "類型 '" in message and "' 在" in message:
                    type_start = message.find("類型 '") + 3
                    type_end = message.find("' 在", type_start)
                    missing_type = message[type_start:type_end]
                    analysis['specific_issues'][f'缺失類型_{missing_type}'].append({
                        'schema': schema,
                        'field': field_name,
                        'languages': languages,
                        'type': missing_type
                    })
                    
            elif '可選字段' in message and '類型標記不正確' in message:
                pattern = '可選字段標記'
                # 記錄具體的可選字段問題
                analysis['specific_issues'][f'可選字段_{schema}_{field_name}'].append({
                    'schema': schema,
                    'field': field_name,
                    'languages': languages,
                    'issue': '類型標記不正確'
                })
                
            else:
                pattern = '其他不匹配'
                
            analysis['by_message_pattern'][pattern].append(warning) 
        
        self.analysis = analysis
        return analysis
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """生成詳細報告"""
        if not self.analysis:
            self.analyze_warnings()
            
        # 統計摘要
        summary = {
            'total_warnings': self.analysis['total_warnings'],
            'by_category': {k: len(v) for k, v in self.analysis['by_category'].items()},
            'by_language': {k: len(v) for k, v in self.analysis['by_language'].items()},
            'by_message_pattern': {k: len(v) for k, v in self.analysis['by_message_pattern'].items()}
        }
        
        # 問題 Schema 排名
        schema_counts = Counter({k: len(v) for k, v in self.analysis['by_schema'].items()})
        top_problematic_schemas = dict(schema_counts.most_common(10))
        
        # 具體問題詳情
        detailed_issues = {}
        for pattern, warnings_list in self.analysis['by_message_pattern'].items():
            if not warnings_list:
                continue
                
            pattern_detail = {
                'count': len(warnings_list),
                'affected_schemas': sorted(list(set(w.get('schema_name', '') for w in warnings_list))),
                'affected_languages': sorted(list(set(lang for w in warnings_list for lang in w.get('languages', [])))),
                'sample_messages': [w.get('message', '') for w in warnings_list[:3]],
                'improvement_suggestions': self._get_improvement_suggestions(pattern)
            }
            
            # 添加具體案例
            if pattern == '類型映射缺失':
                missing_types = defaultdict(int)
                for w in warnings_list:
                    msg = w.get('message', '')
                    if "類型 '" in msg and "' 在" in msg:
                        type_start = msg.find("類型 '") + 3
                        type_end = msg.find("' 在", type_start)
                        missing_type = msg[type_start:type_end]
                        missing_types[missing_type] += 1
                        
                pattern_detail['missing_types'] = dict(Counter(missing_types).most_common(10))
                
            elif pattern == '可選字段標記':
                field_issues = defaultdict(int)
                for w in warnings_list:
                    schema = w.get('schema_name', '')
                    field = w.get('field_name', '')
                    if schema and field:
                        field_issues[f"{schema}.{field}"] += 1
                        
                pattern_detail['problematic_fields'] = dict(Counter(field_issues).most_common(10))
            
            detailed_issues[pattern] = pattern_detail
        
        report = {
            'analysis_metadata': {
                'analysis_date': self.analysis['analysis_date'],
                'report_file': str(self.report_file),
                'total_warnings': summary['total_warnings']
            },
            'summary': summary,
            'top_problematic_schemas': top_problematic_schemas,
            'detailed_issues': detailed_issues,
            'improvement_roadmap': self._generate_improvement_roadmap()
        }
        
        return report
    
    def _get_improvement_suggestions(self, pattern: str) -> List[str]:
        """獲取特定模式的改進建議"""
        suggestions = {
            '類型映射缺失': [
                "在 cross_language_interface.py 中補充完整的類型映射配置",
                "為複合類型 (如 Optional[T], List[T], Dict[K,V]) 添加映射規則",
                "實現動態類型映射生成機制",
                "添加類型映射完整性驗證"
            ],
            '可選字段標記': [
                "統一 YAML Schema 中可選字段的定義格式",
                "改進代碼生成器對可選類型的處理邏輯",
                "建立跨語言可選類型轉換規則",
                "添加可選字段標記驗證機制"
            ],
            '其他不匹配': [
                "統一命名約定處理 (snake_case vs camelCase)",
                "改進註解和文檔生成格式",
                "優化預設值處理機制",
                "建立代碼風格一致性檢查"
            ]
        }
        return suggestions.get(pattern, ["需要進一步分析具體問題"])
    
    def _generate_improvement_roadmap(self) -> Dict[str, Any]:
        """生成改進路線圖"""
        return {
            "Phase 1: 類型映射完善 (高優先級)": {
                "description": "補充缺失的類型映射，解決 337 個類型映射警告",
                "tasks": [
                    "補充 Python 語言的完整類型映射",
                    "添加複合類型映射規則 (Dict[str, Any], List[T] 等)",
                    "實現動態類型映射生成",
                    "建立類型映射驗證機制"
                ],
                "expected_impact": "減少 ~300 個警告"
            },
            "Phase 2: 可選字段標準化 (中優先級)": {
                "description": "統一可選字段處理，解決 352 個可選字段警告",
                "tasks": [
                    "建立統一的可選字段 YAML 定義標準",
                    "改進代碼生成器的可選類型處理邏輯",
                    "添加跨語言可選類型驗證",
                    "建立可選字段最佳實踐文檔"
                ],
                "expected_impact": "減少 ~300 個警告"
            },
            "Phase 3: 代碼品質提升 (低優先級)": {
                "description": "提升整體代碼品質和一致性",
                "tasks": [
                    "統一命名約定處理",
                    "改進註解和文檔生成",
                    "優化預設值處理機制",
                    "建立持續監控機制"
                ],
                "expected_impact": "減少剩餘警告，提升系統品質"
            }
        }
    
    def save_report(self, output_file: str = "detailed_warning_analysis.json") -> bool:
        """保存詳細報告"""
        try:
            report = self.generate_detailed_report()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            print(f"✅ 詳細警告分析報告已保存: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ 保存報告失敗: {e}")
            return False
    
    def print_summary(self):
        """打印摘要"""
        if not self.analysis:
            self.analyze_warnings()
            
        print("\n" + "="*60)
        print("🔍 AIVA 跨語言警告詳細分析摘要")
        print("="*60)
        
        print(f"\n📊 總體統計:")
        print(f"   總警告數量: {self.analysis['total_warnings']} 個")
        print(f"   分析時間: {self.analysis['analysis_date']}")
        
        print(f"\n📝 按類別分布:")
        for pattern, warnings_list in self.analysis['by_message_pattern'].items():
            print(f"   {pattern}: {len(warnings_list)} 個")
            
        print(f"\n🌍 按語言分布:")
        for lang, warnings_list in self.analysis['by_language'].items():
            print(f"   {lang}: {len(warnings_list)} 個")
            
        print(f"\n⚠️  最多警告的 Schema (前5名):")
        schema_counts = Counter({k: len(v) for k, v in self.analysis['by_schema'].items()})
        for schema, count in schema_counts.most_common(5):
            print(f"   {schema}: {count} 個警告")
            
        print(f"\n✨ 改進潛力:")
        print(f"   通過類型映射完善可減少 ~337 個警告")
        print(f"   通過可選字段標準化可減少 ~352 個警告")
        print(f"   預期可將警告數量降至 <100 個")

def main():
    """主函數"""
    analyzer = CrossLanguageWarningAnalyzer()
    
    if not analyzer.load_validation_report():
        sys.exit(1)
        
    analyzer.print_summary()
    analyzer.save_report()
    
    print(f"\n🎯 下一步行動:")
    print(f"   1. 查看 detailed_warning_analysis.json 了解詳細問題")
    print(f"   2. 參考 CROSS_LANGUAGE_WARNING_ANALYSIS.md 制定改進計劃")
    print(f"   3. 優先處理類型映射缺失問題")

if __name__ == "__main__":
    main()