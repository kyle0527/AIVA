#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA Docker 基礎設施分析工具
分析當前 Docker 基礎設施並提供組織建議
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

class DockerInfrastructureAnalyzer:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "repo_path": str(self.repo_path),
            "docker_files": {},
            "architecture_analysis": {},
            "growth_prediction": {},
            "organization_recommendations": []
        }
    
    def analyze_docker_files(self) -> Dict[str, Any]:
        """分析所有 Docker 相關文件"""
        docker_files = {}
        
        # 掃描根目錄的 Docker 文件
        for pattern in ["Dockerfile*", "docker-compose*", "*.dockerfile"]:
            for file_path in self.repo_path.glob(pattern):
                if file_path.is_file():
                    file_info = self._get_file_info(file_path)
                    docker_files[str(file_path.relative_to(self.repo_path))] = file_info
        
        # 掃描 docker/ 子目錄
        docker_dir = self.repo_path / "docker"
        if docker_dir.exists():
            for file_path in docker_dir.rglob("*"):
                if file_path.is_file():
                    file_info = self._get_file_info(file_path)
                    docker_files[str(file_path.relative_to(self.repo_path))] = file_info
        
        # 掃描 k8s/ 目錄
        k8s_dir = self.repo_path / "k8s"
        if k8s_dir.exists():
            for file_path in k8s_dir.rglob("*.yaml"):
                if file_path.is_file():
                    file_info = self._get_file_info(file_path)
                    docker_files[str(file_path.relative_to(self.repo_path))] = file_info
        
        # 掃描 helm/ 目錄
        helm_dir = self.repo_path / "helm"
        if helm_dir.exists():
            for file_path in helm_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    file_info = self._get_file_info(file_path)
                    docker_files[str(file_path.relative_to(self.repo_path))] = file_info
        
        self.analysis_results["docker_files"] = docker_files
        return docker_files
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """獲取文件詳細信息"""
        stat = file_path.stat()
        
        # 計算文件年齡
        file_age_days = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
        
        file_info = {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "age_days": file_age_days,
            "directory": str(file_path.parent.relative_to(self.repo_path)),
            "extension": file_path.suffix,
            "type": self._classify_docker_file(file_path)
        }
        
        # 分析文件內容（如果是文本文件且不太大）
        if stat.st_size < 100000:  # 100KB 以下
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_info["content_analysis"] = self._analyze_file_content(content, file_path)
            except Exception as e:
                file_info["content_analysis"] = {"error": str(e)}
        
        return file_info
    
    def _classify_docker_file(self, file_path: Path) -> str:
        """分類 Docker 文件類型"""
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()
        
        if name.startswith("dockerfile"):
            if "core" in name:
                return "dockerfile_core"
            elif "component" in name:
                return "dockerfile_component"
            elif "patch" in name:
                return "dockerfile_patch"
            elif "minimal" in name:
                return "dockerfile_minimal"
            else:
                return "dockerfile_generic"
        elif name.startswith("docker-compose"):
            return "docker_compose"
        elif name.endswith(".yaml") or name.endswith(".yml"):
            if parent == "k8s":
                return "kubernetes_manifest"
            elif parent == "helm" or "helm" in str(file_path):
                return "helm_chart"
            else:
                return "yaml_config"
        else:
            return "docker_related"
    
    def _analyze_file_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """分析文件內容"""
        analysis = {
            "lines": len(content.splitlines()),
            "size_chars": len(content)
        }
        
        # Dockerfile 特定分析
        if file_path.name.startswith("Dockerfile"):
            analysis.update(self._analyze_dockerfile_content(content))
        
        # docker-compose 特定分析
        elif file_path.name.startswith("docker-compose"):
            analysis.update(self._analyze_compose_content(content))
        
        # Kubernetes 特定分析
        elif file_path.suffix in [".yaml", ".yml"] and "k8s" in str(file_path):
            analysis.update(self._analyze_k8s_content(content))
        
        return analysis
    
    def _analyze_dockerfile_content(self, content: str) -> Dict[str, Any]:
        """分析 Dockerfile 內容"""
        lines = content.splitlines()
        analysis = {
            "from_image": None,
            "exposed_ports": [],
            "copy_commands": 0,
            "run_commands": 0,
            "env_vars": 0,
            "workdir": None,
            "maintainer": None,
            "labels": {}
        }
        
        for line in lines:
            line = line.strip()
            upper_line = line.upper()
            
            if upper_line.startswith("FROM "):
                analysis["from_image"] = line[5:].strip()
            elif upper_line.startswith("EXPOSE "):
                ports = line[7:].strip().split()
                analysis["exposed_ports"].extend(ports)
            elif upper_line.startswith("COPY ") or upper_line.startswith("ADD "):
                analysis["copy_commands"] += 1
            elif upper_line.startswith("RUN "):
                analysis["run_commands"] += 1
            elif upper_line.startswith("ENV "):
                analysis["env_vars"] += 1
            elif upper_line.startswith("WORKDIR "):
                analysis["workdir"] = line[8:].strip()
            elif upper_line.startswith("MAINTAINER "):
                analysis["maintainer"] = line[11:].strip()
            elif upper_line.startswith("LABEL "):
                # 簡單的 label 解析
                label_part = line[6:].strip()
                if "=" in label_part:
                    key, value = label_part.split("=", 1)
                    analysis["labels"][key.strip()] = value.strip().strip('"')
        
        return analysis
    
    def _analyze_compose_content(self, content: str) -> Dict[str, Any]:
        """分析 docker-compose 內容"""
        analysis = {
            "version": None,
            "services_count": 0,
            "networks_count": 0,
            "volumes_count": 0,
            "services": []
        }
        
        # 簡單的 YAML 解析 - 計算服務數量
        lines = content.splitlines()
        in_services = False
        service_indent = None
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith("version:"):
                analysis["version"] = stripped.split(":", 1)[1].strip().strip('"\'')
            elif stripped == "services:":
                in_services = True
                continue
            elif stripped in ["networks:", "volumes:", "configs:", "secrets:"]:
                in_services = False
                if stripped == "networks:":
                    analysis["networks_count"] = self._count_yaml_items(lines, lines.index(line))
                elif stripped == "volumes:":
                    analysis["volumes_count"] = self._count_yaml_items(lines, lines.index(line))
                continue
            
            if in_services and line and not line.startswith(' ') and not line.startswith('#'):
                # 這是一個服務定義
                if ':' in line:
                    service_name = line.split(':')[0].strip()
                    analysis["services"].append(service_name)
                    analysis["services_count"] += 1
        
        return analysis
    
    def _analyze_k8s_content(self, content: str) -> Dict[str, Any]:
        """分析 Kubernetes manifest 內容"""
        analysis = {
            "api_version": None,
            "kind": None,
            "namespace": None,
            "name": None
        }
        
        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith("apiVersion:"):
                analysis["api_version"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("kind:"):
                analysis["kind"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("namespace:"):
                analysis["namespace"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("name:") and analysis["name"] is None:
                analysis["name"] = stripped.split(":", 1)[1].strip()
        
        return analysis
    
    def _count_yaml_items(self, lines: List[str], start_index: int) -> int:
        """計算 YAML 節點下的項目數量"""
        count = 0
        base_indent = len(lines[start_index]) - len(lines[start_index].lstrip())
        
        for i in range(start_index + 1, len(lines)):
            line = lines[i]
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            if current_indent <= base_indent and line.strip():
                break
            
            if current_indent == base_indent + 2 and ':' in line:
                count += 1
        
        return count
    
    def analyze_architecture(self) -> Dict[str, Any]:
        """分析 AIVA 架構和 Docker 使用模式"""
        arch_analysis = {
            "service_architecture": self._analyze_service_architecture(),
            "containerization_strategy": self._analyze_containerization_strategy(),
            "deployment_patterns": self._analyze_deployment_patterns(),
            "file_distribution": self._analyze_file_distribution()
        }
        
        self.analysis_results["architecture_analysis"] = arch_analysis
        return arch_analysis
    
    def _analyze_service_architecture(self) -> Dict[str, Any]:
        """分析服務架構"""
        services_dir = self.repo_path / "services"
        
        analysis = {
            "core_services": [],
            "feature_components": [],
            "total_services": 0
        }
        
        if services_dir.exists():
            for service_dir in services_dir.iterdir():
                if service_dir.is_dir() and not service_dir.name.startswith('.'):
                    service_name = service_dir.name
                    analysis["total_services"] += 1
                    
                    if service_name in ["core", "aiva_common"]:
                        analysis["core_services"].append(service_name)
                    else:
                        analysis["feature_components"].append(service_name)
        
        return analysis
    
    def _analyze_containerization_strategy(self) -> Dict[str, Any]:
        """分析容器化策略"""
        dockerfiles = [f for f in self.analysis_results["docker_files"].keys() 
                      if f.startswith("Dockerfile")]
        
        strategy = {
            "dockerfile_count": len(dockerfiles),
            "dockerfile_types": {},
            "multi_stage_builds": 0,
            "base_images": set(),
            "specialized_containers": []
        }
        
        for dockerfile_path in dockerfiles:
            file_info = self.analysis_results["docker_files"][dockerfile_path]
            file_type = file_info["type"]
            
            if file_type not in strategy["dockerfile_types"]:
                strategy["dockerfile_types"][file_type] = 0
            strategy["dockerfile_types"][file_type] += 1
            
            # 分析內容
            content_analysis = file_info.get("content_analysis", {})
            if "from_image" in content_analysis:
                strategy["base_images"].add(content_analysis["from_image"])
            
            # 檢查是否是特化容器
            if any(keyword in dockerfile_path.lower() for keyword in 
                   ["core", "component", "patch", "minimal"]):
                strategy["specialized_containers"].append(dockerfile_path)
        
        strategy["base_images"] = list(strategy["base_images"])
        return strategy
    
    def _analyze_deployment_patterns(self) -> Dict[str, Any]:
        """分析部署模式"""
        patterns = {
            "has_docker_compose": False,
            "has_kubernetes": False,
            "has_helm": False,
            "deployment_environments": []
        }
        
        # 檢查 docker-compose
        if any(f.startswith("docker-compose") for f in self.analysis_results["docker_files"].keys()):
            patterns["has_docker_compose"] = True
            patterns["deployment_environments"].append("docker-compose")
        
        # 檢查 Kubernetes
        if any("k8s/" in f for f in self.analysis_results["docker_files"].keys()):
            patterns["has_kubernetes"] = True
            patterns["deployment_environments"].append("kubernetes")
        
        # 檢查 Helm
        if any("helm/" in f for f in self.analysis_results["docker_files"].keys()):
            patterns["has_helm"] = True
            patterns["deployment_environments"].append("helm")
        
        return patterns
    
    def _analyze_file_distribution(self) -> Dict[str, Any]:
        """分析文件分佈"""
        distribution = {
            "root_level_files": 0,
            "docker_subdir_files": 0,
            "k8s_files": 0,
            "helm_files": 0,
            "other_locations": 0
        }
        
        for file_path in self.analysis_results["docker_files"].keys():
            if "/" not in file_path:
                distribution["root_level_files"] += 1
            elif file_path.startswith("docker/"):
                distribution["docker_subdir_files"] += 1
            elif file_path.startswith("k8s/"):
                distribution["k8s_files"] += 1
            elif file_path.startswith("helm/"):
                distribution["helm_files"] += 1
            else:
                distribution["other_locations"] += 1
        
        return distribution
    
    def predict_growth(self) -> Dict[str, Any]:
        """預測 Docker 基礎設施增長"""
        prediction = {
            "growth_factors": [],
            "expected_dockerfile_growth": "moderate",
            "recommended_organization_threshold": 8,
            "current_complexity_score": 0,
            "projected_complexity_score": 0
        }
        
        # 分析增長因子
        arch_analysis = self.analysis_results["architecture_analysis"]
        
        # 服務數量因子
        total_services = arch_analysis["service_architecture"]["total_services"]
        if total_services > 3:
            prediction["growth_factors"].append(f"多服務架構 ({total_services} 個服務)")
        
        # 特化容器因子
        specialized_count = len(arch_analysis["containerization_strategy"]["specialized_containers"])
        if specialized_count > 2:
            prediction["growth_factors"].append(f"多種特化容器 ({specialized_count} 個)")
        
        # 部署環境因子
        env_count = len(arch_analysis["deployment_patterns"]["deployment_environments"])
        if env_count > 1:
            prediction["growth_factors"].append(f"多部署環境 ({env_count} 個)")
        
        # 計算複雜度分數
        current_complexity = (
            len(self.analysis_results["docker_files"]) * 1 +
            specialized_count * 2 +
            env_count * 3 +
            total_services * 1
        )
        
        prediction["current_complexity_score"] = current_complexity
        prediction["projected_complexity_score"] = current_complexity * 1.5  # 預期增長50%
        
        # 預測增長程度
        if current_complexity < 10:
            prediction["expected_dockerfile_growth"] = "低"
        elif current_complexity < 20:
            prediction["expected_dockerfile_growth"] = "中等"
        else:
            prediction["expected_dockerfile_growth"] = "高"
        
        self.analysis_results["growth_prediction"] = prediction
        return prediction
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成組織建議"""
        recommendations = []
        
        # 基於當前複雜度的建議
        complexity_score = self.analysis_results["growth_prediction"]["current_complexity_score"]
        file_distribution = self.analysis_results["architecture_analysis"]["file_distribution"]
        
        # 建議1: 文件組織
        if file_distribution["root_level_files"] > 4:
            recommendations.append({
                "priority": "高",
                "category": "文件組織",
                "title": "整理根目錄 Docker 文件",
                "description": f"根目錄有 {file_distribution['root_level_files']} 個 Docker 文件，建議移至專門目錄",
                "action": "創建 docker/ 子目錄並移動文件",
                "estimated_effort": "1-2 小時"
            })
        
        # 建議2: Dockerfile 標準化
        dockerfile_types = self.analysis_results["architecture_analysis"]["containerization_strategy"]["dockerfile_types"]
        if len(dockerfile_types) > 3:
            recommendations.append({
                "priority": "中",
                "category": "標準化",
                "title": "建立 Dockerfile 命名規範",
                "description": f"發現 {len(dockerfile_types)} 種不同類型的 Dockerfile，需要統一命名規範",
                "action": "制定並應用一致的命名模式",
                "estimated_effort": "2-3 小時"
            })
        
        # 建議3: 部署配置整理
        if self.analysis_results["architecture_analysis"]["deployment_patterns"]["has_kubernetes"]:
            recommendations.append({
                "priority": "中",
                "category": "部署配置",
                "title": "Kubernetes 配置版本管理",
                "description": "建立 K8s 配置的版本管理和環境區分",
                "action": "按環境組織 K8s manifests (dev/staging/prod)",
                "estimated_effort": "3-4 小時"
            })
        
        # 建議4: 增長預備
        if complexity_score > 15:
            recommendations.append({
                "priority": "中",
                "category": "架構準備",
                "title": "準備容器化架構擴展",
                "description": "系統複雜度較高，建議建立容器化治理框架",
                "action": "建立 Docker 文件模板和最佳實踐文檔",
                "estimated_effort": "4-6 小時"
            })
        
        # 建議5: 自動化
        if len(self.analysis_results["docker_files"]) > 10:
            recommendations.append({
                "priority": "低",
                "category": "自動化",
                "title": "建立容器構建自動化",
                "description": "Docker 文件數量較多，建議自動化構建流程",
                "action": "設置 CI/CD 管道自動構建和推送容器",
                "estimated_effort": "6-8 小時"
            })
        
        # 建議6: 文檔化
        recommendations.append({
            "priority": "中",
            "category": "文檔",
            "title": "更新容器化文檔",
            "description": "確保所有 Docker 文件都有適當的文檔說明",
            "action": "為每個 Dockerfile 添加詳細註釋和使用說明",
            "estimated_effort": "2-3 小時"
        })
        
        self.analysis_results["organization_recommendations"] = recommendations
        return recommendations
    
    def generate_report(self) -> str:
        """生成完整的分析報告"""
        report_lines = [
            "# AIVA Docker 基礎設施分析報告",
            "",
            f"**分析時間**: {self.analysis_results['timestamp']}",
            f"**分析路徑**: {self.analysis_results['repo_path']}",
            "",
            "## 📊 總體概況",
            "",
            f"- **Docker 相關文件總數**: {len(self.analysis_results['docker_files'])}",
            f"- **複雜度評分**: {self.analysis_results['growth_prediction']['current_complexity_score']}/100",
            f"- **預期增長**: {self.analysis_results['growth_prediction']['expected_dockerfile_growth']}",
            "",
            "## 📁 文件分佈分析",
            ""
        ]
        
        # 文件分佈表格
        distribution = self.analysis_results["architecture_analysis"]["file_distribution"]
        report_lines.extend([
            "| 位置 | 文件數量 | 說明 |",
            "|------|----------|------|",
            f"| 根目錄 | {distribution['root_level_files']} | 直接在專案根目錄 |",
            f"| docker/ | {distribution['docker_subdir_files']} | Docker 專用目錄 |",
            f"| k8s/ | {distribution['k8s_files']} | Kubernetes 配置 |",
            f"| helm/ | {distribution['helm_files']} | Helm Charts |",
            f"| 其他位置 | {distribution['other_locations']} | 其他子目錄 |",
            "",
            "## 🐳 Docker 文件詳細分析",
            ""
        ])
        
        # Docker 文件列表
        for file_path, file_info in self.analysis_results["docker_files"].items():
            age_status = "🟢 最新" if file_info["age_days"] < 7 else "🟡 一週前" if file_info["age_days"] < 14 else "🔴 兩週前"
            report_lines.append(f"### {file_path}")
            report_lines.extend([
                f"- **類型**: {file_info['type']}",
                f"- **大小**: {file_info['size']} bytes",
                f"- **最後修改**: {file_info['last_modified']} ({age_status})",
                f"- **文件年齡**: {file_info['age_days']} 天",
                ""
            ])
            
            # 內容分析
            if "content_analysis" in file_info and "error" not in file_info["content_analysis"]:
                content = file_info["content_analysis"]
                if file_info["type"].startswith("dockerfile"):
                    report_lines.extend([
                        f"  - **基礎映像**: {content.get('from_image', 'N/A')}",
                        f"  - **暴露端口**: {', '.join(content.get('exposed_ports', [])) or 'None'}",
                        f"  - **複製命令**: {content.get('copy_commands', 0)}",
                        f"  - **執行命令**: {content.get('run_commands', 0)}",
                        ""
                    ])
                elif file_info["type"] == "docker_compose":
                    report_lines.extend([
                        f"  - **版本**: {content.get('version', 'N/A')}",
                        f"  - **服務數量**: {content.get('services_count', 0)}",
                        f"  - **網絡數量**: {content.get('networks_count', 0)}",
                        f"  - **數據卷數量**: {content.get('volumes_count', 0)}",
                        ""
                    ])
        
        # 架構分析
        arch = self.analysis_results["architecture_analysis"]
        report_lines.extend([
            "## 🏗️ 架構分析",
            "",
            f"- **總服務數**: {arch['service_architecture']['total_services']}",
            f"- **核心服務**: {', '.join(arch['service_architecture']['core_services'])}",
            f"- **功能組件**: {', '.join(arch['service_architecture']['feature_components'])}",
            "",
            f"- **Dockerfile 數量**: {arch['containerization_strategy']['dockerfile_count']}",
            f"- **基礎映像**: {', '.join(arch['containerization_strategy']['base_images'])}",
            f"- **特化容器**: {len(arch['containerization_strategy']['specialized_containers'])}",
            "",
            f"- **部署環境**: {', '.join(arch['deployment_patterns']['deployment_environments'])}",
            ""
        ])
        
        # 增長預測
        growth = self.analysis_results["growth_prediction"]
        report_lines.extend([
            "## 📈 增長預測",
            "",
            f"- **當前複雜度**: {growth['current_complexity_score']}/100",
            f"- **預期複雜度**: {growth['projected_complexity_score']}/100",
            f"- **預期增長程度**: {growth['expected_dockerfile_growth']}",
            "",
            "**增長因子**:",
        ])
        
        for factor in growth["growth_factors"]:
            report_lines.append(f"- {factor}")
        
        # 組織建議
        report_lines.extend([
            "",
            "## 💡 組織建議",
            ""
        ])
        
        for i, rec in enumerate(self.analysis_results["organization_recommendations"], 1):
            priority_emoji = {"高": "🔴", "中": "🟡", "低": "🟢"}
            report_lines.extend([
                f"### {i}. {rec['title']} {priority_emoji.get(rec['priority'], '⚪')}",
                "",
                f"**優先級**: {rec['priority']}",
                f"**類別**: {rec['category']}",
                f"**描述**: {rec['description']}",
                f"**建議行動**: {rec['action']}",
                f"**預估工作量**: {rec['estimated_effort']}",
                ""
            ])
        
        # 結論和下一步
        report_lines.extend([
            "## 🎯 結論和下一步",
            "",
            "基於分析結果，建議按以下順序進行組織改進：",
            "",
            "1. **立即執行** (高優先級建議)",
            "2. **短期規劃** (中優先級建議)",
            "3. **長期維護** (低優先級建議)",
            "",
            "這樣的組織方式將有助於：",
            "- 提高 Docker 文件的可維護性",
            "- 減少部署配置的複雜度",
            "- 為未來的架構擴展做好準備",
            "- 建立標準化的容器化流程",
            "",
            f"---",
            f"*報告生成時間: {datetime.now().isoformat()}*"
        ])
        
        return "\n".join(report_lines)
    
    def save_report(self, output_file: str = None) -> str:
        """保存分析報告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"docker_infrastructure_analysis_{timestamp}.md"
        
        output_path = self.repo_path / output_file
        
        # 生成報告內容
        report_content = self.generate_report()
        
        # 添加時間戳標頭
        timestamped_content = f"""---
Created: {datetime.now().strftime("%Y-%m-%d")}
Last Modified: {datetime.now().strftime("%Y-%m-%d")}
Document Type: Analysis Report
Analysis Target: Docker Infrastructure
---

{report_content}
"""
        
        # 保存報告
        output_path.write_text(timestamped_content, encoding='utf-8')
        
        # 保存 JSON 數據
        json_file = output_path.with_suffix('.json')
        json_file.write_text(json.dumps(self.analysis_results, indent=2, ensure_ascii=False), 
                           encoding='utf-8')
        
        return str(output_path)
    
    def run_full_analysis(self) -> str:
        """執行完整分析並生成報告"""
        print("🔍 開始 Docker 基礎設施分析...")
        
        # 1. 分析 Docker 文件
        print("📁 分析 Docker 相關文件...")
        self.analyze_docker_files()
        print(f"   發現 {len(self.analysis_results['docker_files'])} 個文件")
        
        # 2. 分析架構
        print("🏗️ 分析服務架構...")
        self.analyze_architecture()
        
        # 3. 預測增長
        print("📈 分析增長趨勢...")
        self.predict_growth()
        
        # 4. 生成建議
        print("💡 生成組織建議...")
        self.generate_recommendations()
        print(f"   生成 {len(self.analysis_results['organization_recommendations'])} 條建議")
        
        # 5. 保存報告
        print("📄 生成分析報告...")
        report_path = self.save_report()
        
        print(f"✅ 分析完成！報告已保存至: {report_path}")
        return report_path

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Docker 基礎設施分析工具")
    parser.add_argument("--repo-path", default=".", help="專案路徑 (預設: 當前目錄)")
    parser.add_argument("--output", help="輸出報告文件名")
    parser.add_argument("--json-only", action="store_true", help="只輸出 JSON 格式")
    
    args = parser.parse_args()
    
    # 創建分析器
    analyzer = DockerInfrastructureAnalyzer(args.repo_path)
    
    if args.json_only:
        # 只執行分析，輸出 JSON
        analyzer.analyze_docker_files()
        analyzer.analyze_architecture()
        analyzer.predict_growth()
        analyzer.generate_recommendations()
        
        print(json.dumps(analyzer.analysis_results, indent=2, ensure_ascii=False))
    else:
        # 執行完整分析並生成報告
        report_path = analyzer.run_full_analysis()
        print(f"\n📊 分析摘要:")
        print(f"   - Docker 文件數量: {len(analyzer.analysis_results['docker_files'])}")
        print(f"   - 複雜度評分: {analyzer.analysis_results['growth_prediction']['current_complexity_score']}")
        print(f"   - 建議數量: {len(analyzer.analysis_results['organization_recommendations'])}")
        print(f"   - 報告位置: {report_path}")

if __name__ == "__main__":
    main()