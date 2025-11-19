#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA Docker 基礎設施更新與組織工具
根據 aiva_common 指南和分析報告，對 Docker 基礎設施進行更新和重組
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class DockerInfrastructureUpdater:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.backup_dir = self.repo_path / "_backup_docker" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.update_log = []
        
        # 基於分析報告的現狀
        self.current_state = {
            "total_docker_files": 18,
            "complexity_score": 35,
            "root_level_files": 18,
            "docker_subdir_files": 0,
            "growth_prediction": "高"
        }
        
        # 基於 AIVA 架構的組織計劃
        self.organization_plan = {
            "docker/": {
                "description": "Docker 基礎設施統一目錄",
                "subdirs": {
                    "core/": "核心服務容器配置",
                    "components/": "功能組件容器配置", 
                    "infrastructure/": "基礎設施服務配置",
                    "compose/": "Docker Compose 配置文件",
                    "k8s/": "Kubernetes 配置",
                    "helm/": "Helm Charts"
                }
            }
        }
    
    def log_update(self, operation: str, details: str, success: bool = True):
        """記錄更新操作"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "details": details,
            "success": success
        }
        self.update_log.append(log_entry)
        status = "✅" if success else "❌"
        print(f"{status} {operation}: {details}")
    
    def create_backup(self):
        """創建當前 Docker 配置的備份"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_files = []
        
        # 備份根目錄的 Docker 文件
        for pattern in ["Dockerfile*", "docker-compose*", "*.dockerfile"]:
            for file_path in self.repo_path.glob(pattern):
                if file_path.is_file():
                    backup_path = self.backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    backup_files.append(str(file_path.relative_to(self.repo_path)))
        
        # 備份現有的 docker/ 目錄
        docker_dir = self.repo_path / "docker"
        if docker_dir.exists():
            backup_docker_dir = self.backup_dir / "docker"
            shutil.copytree(docker_dir, backup_docker_dir)
            for file_path in docker_dir.rglob("*"):
                if file_path.is_file():
                    backup_files.append(str(file_path.relative_to(self.repo_path)))
        
        # 備份 k8s/ 和 helm/ 目錄
        for dir_name in ["k8s", "helm"]:
            source_dir = self.repo_path / dir_name
            if source_dir.exists():
                backup_target = self.backup_dir / dir_name
                shutil.copytree(source_dir, backup_target)
                for file_path in source_dir.rglob("*"):
                    if file_path.is_file():
                        backup_files.append(str(file_path.relative_to(self.repo_path)))
        
        self.log_update("創建備份", f"備份了 {len(backup_files)} 個文件到 {self.backup_dir}")
        return backup_files
    
    def analyze_current_docker_structure(self) -> Dict[str, Any]:
        """分析當前 Docker 結構"""
        analysis = {
            "root_dockerfiles": [],
            "compose_files": [],
            "k8s_files": [],
            "helm_files": [],
            "docker_subdir_files": [],
            "service_mapping": {}
        }
        
        # 掃描根目錄 Docker 文件
        for file_path in self.repo_path.glob("Dockerfile*"):
            if file_path.is_file():
                analysis["root_dockerfiles"].append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "type": self._classify_dockerfile(file_path.name)
                })
        
        # 掃描 compose 文件
        for file_path in self.repo_path.glob("docker-compose*"):
            if file_path.is_file():
                analysis["compose_files"].append({
                    "name": file_path.name,
                    "path": str(file_path)
                })
        
        # 掃描 k8s 目錄
        k8s_dir = self.repo_path / "k8s"
        if k8s_dir.exists():
            for file_path in k8s_dir.rglob("*.yaml"):
                if file_path.is_file():
                    analysis["k8s_files"].append({
                        "name": file_path.name,
                        "path": str(file_path)
                    })
        
        # 掃描 helm 目錄
        helm_dir = self.repo_path / "helm"
        if helm_dir.exists():
            for file_path in helm_dir.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    analysis["helm_files"].append({
                        "name": file_path.name,
                        "path": str(file_path)
                    })
        
        # 掃描現有 docker 子目錄
        docker_dir = self.repo_path / "docker"
        if docker_dir.exists():
            for file_path in docker_dir.rglob("*"):
                if file_path.is_file():
                    analysis["docker_subdir_files"].append({
                        "name": file_path.name,
                        "path": str(file_path)
                    })
        
        # 基於 aiva_common 指南分析服務映射
        analysis["service_mapping"] = self._analyze_service_mapping()
        
        return analysis
    
    def _classify_dockerfile(self, filename: str) -> str:
        """根據文件名分類 Dockerfile"""
        name = filename.lower()
        if "core" in name:
            return "core_service"
        elif "component" in name:
            return "component_service"
        elif "patch" in name:
            return "patch_update"
        elif "minimal" in name:
            return "minimal_build"
        else:
            return "generic"
    
    def _analyze_service_mapping(self) -> Dict[str, Any]:
        """基於 aiva_common 指南分析服務映射"""
        services_dir = self.repo_path / "services"
        mapping = {
            "core_services": [],
            "feature_components": [],
            "infrastructure_services": [],
            "service_count": 0
        }
        
        if services_dir.exists():
            for service_dir in services_dir.iterdir():
                if service_dir.is_dir() and not service_dir.name.startswith('.'):
                    service_name = service_dir.name
                    mapping["service_count"] += 1
                    
                    if service_name in ["core", "aiva_common"]:
                        mapping["core_services"].append(service_name)
                    elif service_name == "features":
                        mapping["feature_components"].append(service_name)
                        # 分析 features 子服務
                        features_dir = service_dir
                        if features_dir.exists():
                            for feature_dir in features_dir.iterdir():
                                if feature_dir.is_dir() and not feature_dir.name.startswith('.'):
                                    if feature_dir.name != "common":
                                        mapping["feature_components"].append(f"features/{feature_dir.name}")
                    elif service_name in ["scan", "integration"]:
                        mapping["infrastructure_services"].append(service_name)
        
        return mapping
    
    def create_new_docker_structure(self):
        """創建新的 Docker 目錄結構"""
        new_docker_dir = self.repo_path / "docker"
        
        # 創建主目錄結構
        for subdir, description in self.organization_plan["docker/"]["subdirs"].items():
            target_dir = new_docker_dir / subdir.rstrip('/')
            target_dir.mkdir(parents=True, exist_ok=True)
            self.log_update("創建目錄", f"{target_dir}: {description}")
            
            # 創建 README.md
            readme_content = f"""# {subdir.rstrip('/').title()} Docker 配置

{description}

## 文件說明

此目錄包含 {description.lower()} 的相關配置文件。

## 使用方式

```bash
# 建構映像
docker build -f {subdir}Dockerfile.xxx .

# 使用 docker-compose
docker-compose -f {subdir}docker-compose.xxx.yml up
```

---
Created: {datetime.now().strftime('%Y-%m-%d')}
Last Modified: {datetime.now().strftime('%Y-%m-%d')}
"""
            readme_path = target_dir / "README.md"
            readme_path.write_text(readme_content, encoding='utf-8')
    
    def reorganize_docker_files(self):
        """重組 Docker 文件"""
        analysis = self.analyze_current_docker_structure()
        new_docker_dir = self.repo_path / "docker"
        
        # 移動 Dockerfile 文件
        for dockerfile in analysis["root_dockerfiles"]:
            source_path = Path(dockerfile["path"])
            
            if dockerfile["type"] == "core_service":
                target_dir = new_docker_dir / "core"
            elif dockerfile["type"] == "component_service":
                target_dir = new_docker_dir / "components"
            elif dockerfile["type"] in ["patch_update", "minimal_build"]:
                target_dir = new_docker_dir / "core"  # 核心服務的變體
            else:
                target_dir = new_docker_dir / "infrastructure"
            
            target_path = target_dir / source_path.name
            if source_path.exists():
                shutil.move(str(source_path), str(target_path))
                self.log_update("移動文件", f"{source_path.name} → {target_path}")
        
        # 移動 docker-compose 文件
        compose_dir = new_docker_dir / "compose"
        for compose_file in analysis["compose_files"]:
            source_path = Path(compose_file["path"])
            target_path = compose_dir / source_path.name
            if source_path.exists():
                shutil.move(str(source_path), str(target_path))
                self.log_update("移動文件", f"{source_path.name} → {target_path}")
        
        # 移動 k8s 文件
        if analysis["k8s_files"]:
            k8s_target_dir = new_docker_dir / "k8s"
            k8s_source_dir = self.repo_path / "k8s"
            if k8s_source_dir.exists():
                # 如果目標目錄已存在，先移除
                if k8s_target_dir.exists():
                    shutil.rmtree(k8s_target_dir)
                shutil.move(str(k8s_source_dir), str(k8s_target_dir))
                self.log_update("移動目錄", f"k8s/ → docker/k8s/")
        
        # 移動 helm 文件
        if analysis["helm_files"]:
            helm_target_dir = new_docker_dir / "helm"
            helm_source_dir = self.repo_path / "helm"
            if helm_source_dir.exists():
                # 如果目標目錄已存在，先移除
                if helm_target_dir.exists():
                    shutil.rmtree(helm_target_dir)
                shutil.move(str(helm_source_dir), str(helm_target_dir))
                self.log_update("移動目錄", f"helm/ → docker/helm/")
        
        # 合併現有 docker/ 目錄內容
        old_docker_dir = self.repo_path / "docker_old"
        current_docker_dir = self.repo_path / "docker"
        
        if analysis["docker_subdir_files"]:
            # 暫時重命名現有 docker 目錄
            if current_docker_dir.exists():
                current_docker_dir.rename(old_docker_dir)
                self.log_update("暫存目錄", "暫時重命名現有 docker/ 為 docker_old/")
            
            # 重新創建結構
            self.create_new_docker_structure()
            
            # 合併舊文件
            if old_docker_dir.exists():
                for file_info in analysis["docker_subdir_files"]:
                    source_path = Path(file_info["path"].replace("docker/", "docker_old/"))
                    
                    # 根據文件類型決定目標位置
                    if "compose" in file_info["name"]:
                        target_dir = new_docker_dir / "compose"
                    elif file_info["name"].endswith(('.yaml', '.yml')):
                        target_dir = new_docker_dir / "k8s"
                    else:
                        target_dir = new_docker_dir / "infrastructure"
                    
                    target_path = target_dir / file_info["name"]
                    if source_path.exists():
                        shutil.copy2(str(source_path), str(target_path))
                        self.log_update("合併文件", f"{source_path} → {target_path}")
                
                # 清理舊目錄
                shutil.rmtree(old_docker_dir)
                self.log_update("清理目錄", "移除 docker_old/ 目錄")
    
    def update_documentation(self):
        """更新相關文檔"""
        # 更新主 README
        self._update_main_readme()
        
        # 創建 Docker 使用指南
        self._create_docker_guide()
        
        # 更新 aiva_common 相關文檔
        self._update_aiva_common_references()
    
    def _update_main_readme(self):
        """更新主 README 中的 Docker 說明"""
        main_readme = self.repo_path / "README.md"
        if main_readme.exists():
            content = main_readme.read_text(encoding='utf-8')
            
            # 添加 Docker 結構說明
            docker_section = """
## 🐳 Docker 基礎設施

AIVA 採用統一的 Docker 基礎設施管理，所有容器化配置位於 `docker/` 目錄：

```
docker/
├── core/                    # 核心服務容器
│   ├── Dockerfile.core      # 核心 AI 服務
│   ├── Dockerfile.core.minimal
│   └── Dockerfile.patch
├── components/              # 功能組件容器
│   └── Dockerfile.component
├── infrastructure/          # 基礎設施服務
├── compose/                 # Docker Compose 配置
│   ├── docker-compose.yml
│   └── docker-compose.production.yml
├── k8s/                     # Kubernetes 配置
│   ├── 00-namespace.yaml
│   ├── 01-configmap.yaml
│   ├── 02-storage.yaml
│   ├── 10-core-deployment.yaml
│   └── 20-components-jobs.yaml
└── helm/                    # Helm Charts
    └── aiva/
```

### 快速啟動

```bash
# 啟動完整系統
docker-compose -f docker/compose/docker-compose.yml up -d

# 只啟動核心服務
docker-compose -f docker/compose/docker-compose.yml up -d aiva-core

# Kubernetes 部署
kubectl apply -f docker/k8s/
```

"""
            
            # 如果沒有 Docker 章節，添加到文件末尾
            if "Docker 基礎設施" not in content:
                content += docker_section
                main_readme.write_text(content, encoding='utf-8')
                self.log_update("更新文檔", "主 README.md 添加 Docker 章節")
    
    def _create_docker_guide(self):
        """創建詳細的 Docker 使用指南"""
        guide_content = f"""---
Created: {datetime.now().strftime('%Y-%m-%d')}
Last Modified: {datetime.now().strftime('%Y-%m-%d')}
Document Type: Docker 基礎設施指南
---

# AIVA Docker 基礎設施使用指南

本指南基於 Docker 基礎設施分析報告和 aiva_common 標準編寫。

## 📊 當前狀態

- **Docker 文件總數**: {self.current_state['total_docker_files']}
- **複雜度評分**: {self.current_state['complexity_score']}/100
- **增長預測**: {self.current_state['growth_prediction']}
- **重組狀態**: ✅ 已完成

## 🏗️ 架構概覽

AIVA 採用微服務架構，基於以下容器化策略：

### 核心服務 (永遠運行)
- **aiva-core**: 核心 AI 服務，包含對話助理、經驗管理器
- **基礎設施**: PostgreSQL, Redis, RabbitMQ, Neo4j

### 功能組件 (按需啟動)
- **22個功能組件**: 各種掃描器、測試工具、分析器
- **動態調度**: 根據任務需求啟動相應組件

## 📁 目錄結構說明

```
docker/
├── core/                    # 核心服務容器配置
│   ├── Dockerfile.core      # 主要核心服務
│   ├── Dockerfile.core.minimal  # 最小化版本
│   └── Dockerfile.patch     # 增量更新版本
│
├── components/              # 功能組件容器配置
│   └── Dockerfile.component # 通用組件容器
│
├── infrastructure/          # 基礎設施服務配置
│   └── (未來擴展用)
│
├── compose/                 # Docker Compose 配置
│   ├── docker-compose.yml  # 主要配置
│   └── docker-compose.production.yml  # 生產環境配置
│
├── k8s/                     # Kubernetes 部署配置
│   ├── 00-namespace.yaml   # 命名空間
│   ├── 01-configmap.yaml   # 配置管理
│   ├── 02-storage.yaml     # 存儲配置
│   ├── 10-core-deployment.yaml     # 核心服務部署
│   └── 20-components-jobs.yaml     # 組件任務配置
│
└── helm/                    # Helm Charts
    └── aive/
        ├── Chart.yaml
        └── values.yaml
```

## 🚀 使用方式

### 開發環境

```bash
# 啟動完整開發環境
docker-compose -f docker/compose/docker-compose.yml up -d

# 只啟動基礎設施
docker-compose -f docker/compose/docker-compose.yml up -d postgres redis rabbitmq neo4j

# 啟動核心服務
docker-compose -f docker/compose/docker-compose.yml up -d aiva-core

# 查看服務狀態
docker-compose -f docker/compose/docker-compose.yml ps
```

### 生產環境

```bash
# 使用生產配置
docker-compose -f docker/compose/docker-compose.production.yml up -d

# Kubernetes 部署
kubectl apply -f docker/k8s/

# Helm 部署
helm install aiva docker/helm/aiva/
```

### 單獨構建映像

```bash
# 構建核心服務
docker build -f docker/core/Dockerfile.core -t aiva-core:latest .

# 構建功能組件
docker build -f docker/components/Dockerfile.component -t aiva-component:latest .

# 構建最小化版本
docker build -f docker/core/Dockerfile.core.minimal -t aiva-core:minimal .
```

## 🔧 配置說明

### 環境變量

核心服務支援以下環境變量配置：

```bash
# 模式配置
ENVIRONMENT=docker

# 數據庫配置 (使用單一 DATABASE_URL)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/aiva_db

# RabbitMQ 配置 (使用單一 RABBITMQ_URL)
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
```

### 端口映射

| 服務 | 內部端口 | 外部端口 | 說明 |
|------|---------|---------|------|
| AIVA Core | 8000 | 8000 | 主 API |
| AIVA Core | 8001 | 8001 | 管理 API |
| AIVA Core | 8002 | 8002 | WebSocket |
| PostgreSQL | 5432 | 5432 | 數據庫 |
| Redis | 6379 | 6379 | 緩存 |
| RabbitMQ | 5672 | 5672 | 消息隊列 |
| RabbitMQ UI | 15672 | 15672 | 管理界面 |
| Neo4j | 7687 | 7687 | 圖數據庫 |
| Neo4j UI | 7474 | 7474 | 管理界面 |

## 🔍 故障排除

### 常見問題

1. **容器啟動失敗**
   ```bash
   # 檢查日誌
   docker-compose -f docker/compose/docker-compose.yml logs aiva-core
   
   # 檢查資源使用
   docker stats
   ```

2. **服務連接問題**
   ```bash
   # 檢查網絡連通性
   docker-compose -f docker/compose/docker-compose.yml exec aiva-core ping postgres
   
   # 檢查端口
   docker-compose -f docker/compose/docker-compose.yml port aiva-core 8000
   ```

3. **數據持久化問題**
   ```bash
   # 檢查數據卷
   docker volume ls
   docker volume inspect aiva-git_postgres-data
   ```

### 性能優化

1. **資源限制**
   - 核心服務: 2GB RAM, 1 CPU
   - 功能組件: 1GB RAM, 0.5 CPU
   - 基礎設施: 根據負載調整

2. **網絡優化**
   - 使用內部網絡通信
   - 啟用連接池
   - 配置健康檢查

## 📚 相關文檔

- [AIVA 架構文檔](../reports/architecture/ARCHITECTURE_SUMMARY.md)
- [Docker 基礎設施分析報告](../docker_infrastructure_analysis_20251030_113318.md)
- [aiva_common 開發指南](../services/aiva_common/README.md)

## 🔄 版本記錄

- **2025-10-30**: 初始版本，基於基礎設施分析報告創建
- **未來計劃**: 支援更多部署模式，優化容器大小

---

*最後更新: {datetime.now().isoformat()}*
"""
        
        guide_path = self.repo_path / "docker" / "DOCKER_GUIDE.md"
        guide_path.write_text(guide_content, encoding='utf-8')
        self.log_update("創建文檔", f"Docker 使用指南: {guide_path}")
    
    def _update_aiva_common_references(self):
        """更新 aiva_common 中的 Docker 相關引用"""
        # 更新 continuous_components_sot.json 中的 Docker 配置
        sot_file = self.repo_path / "services" / "aiva_common" / "continuous_components_sot.json"
        if sot_file.exists():
            try:
                with open(sot_file, 'r', encoding='utf-8') as f:
                    sot_data = json.load(f)
                
                # 更新 Docker 整合配置
                if "integration_points" in sot_data and "docker_integration" in sot_data["integration_points"]:
                    docker_config = sot_data["integration_points"]["docker_integration"]
                    docker_config.update({
                        "enabled": True,
                        "docker_socket": "/var/run/docker.sock",
                        "container_health_check": True,
                        "auto_container_restart": True,
                        "config_directory": "docker/",
                        "compose_files": {
                            "development": "docker/compose/docker-compose.yml",
                            "production": "docker/compose/docker-compose.production.yml"
                        },
                        "k8s_directory": "docker/k8s/",
                        "helm_chart": "docker/helm/aiva/"
                    })
                
                # 保存更新
                with open(sot_file, 'w', encoding='utf-8') as f:
                    json.dump(sot_data, f, indent=2, ensure_ascii=False)
                
                self.log_update("更新配置", "aiva_common Docker 整合配置已更新")
                
            except Exception as e:
                self.log_update("更新配置", f"更新 aiva_common 配置失敗: {e}", False)
    
    def validate_new_structure(self) -> Dict[str, Any]:
        """驗證新的目錄結構"""
        docker_dir = self.repo_path / "docker"
        validation = {
            "structure_exists": docker_dir.exists(),
            "subdirs_created": {},
            "files_moved": {},
            "documentation_created": {},
            "issues": []
        }
        
        # 檢查子目錄
        for subdir in self.organization_plan["docker/"]["subdirs"].keys():
            subdir_path = docker_dir / subdir.rstrip('/')
            validation["subdirs_created"][subdir] = subdir_path.exists()
            
            if not subdir_path.exists():
                validation["issues"].append(f"缺少目錄: {subdir}")
        
        # 檢查文件是否正確移動
        expected_files = {
            "core/": ["Dockerfile.core", "Dockerfile.core.minimal", "Dockerfile.patch"],
            "components/": ["Dockerfile.component"],
            "compose/": ["docker-compose.yml", "docker-compose.production.yml"],
            "k8s/": ["00-namespace.yaml", "01-configmap.yaml", "02-storage.yaml", 
                    "10-core-deployment.yaml", "20-components-jobs.yaml"],
            "helm/": ["aiva/Chart.yaml", "aiva/values.yaml"]
        }
        
        for subdir, files in expected_files.items():
            subdir_path = docker_dir / subdir.rstrip('/')
            validation["files_moved"][subdir] = []
            
            for file_name in files:
                file_path = subdir_path / file_name
                if file_path.exists():
                    validation["files_moved"][subdir].append(file_name)
                else:
                    validation["issues"].append(f"缺少文件: {subdir}{file_name}")
        
        # 檢查文檔
        docs_to_check = ["DOCKER_GUIDE.md", "README.md"]
        for doc in docs_to_check:
            doc_path = docker_dir / doc
            validation["documentation_created"][doc] = doc_path.exists()
            
            if not doc_path.exists():
                validation["issues"].append(f"缺少文檔: {doc}")
        
        return validation
    
    def generate_update_report(self) -> str:
        """生成更新報告"""
        validation = self.validate_new_structure()
        
        report_lines = [
            "# AIVA Docker 基礎設施更新報告",
            "",
            f"**更新時間**: {datetime.now().isoformat()}",
            f"**更新路徑**: {self.repo_path}",
            "",
            "## 📊 更新概覽",
            "",
            f"- **原始狀態**: {self.current_state['total_docker_files']} 個文件散布在根目錄",
            f"- **複雜度評分**: {self.current_state['complexity_score']}/100",
            f"- **更新操作數**: {len(self.update_log)}",
            f"- **備份位置**: {self.backup_dir}",
            "",
            "## 🏗️ 新目錄結構",
            "",
            "```",
            "docker/",
        ]
        
        # 添加目錄結構
        for subdir, description in self.organization_plan["docker/"]["subdirs"].items():
            status = "✅" if validation["subdirs_created"].get(subdir, False) else "❌"
            report_lines.append(f"├── {subdir:<20} {status} {description}")
        
        report_lines.extend([
            "```",
            "",
            "## 📁 文件移動結果",
            ""
        ])
        
        # 添加文件移動結果
        for subdir, files in validation["files_moved"].items():
            if files:
                report_lines.append(f"### {subdir}")
                for file_name in files:
                    report_lines.append(f"- ✅ {file_name}")
                report_lines.append("")
        
        # 添加操作日誌
        report_lines.extend([
            "## 📝 更新操作日誌",
            ""
        ])
        
        for log_entry in self.update_log:
            status = "✅" if log_entry["success"] else "❌"
            timestamp = datetime.fromisoformat(log_entry["timestamp"]).strftime("%H:%M:%S")
            report_lines.append(f"- {status} **{timestamp}** {log_entry['operation']}: {log_entry['details']}")
        
        # 添加驗證結果
        report_lines.extend([
            "",
            "## 🔍 驗證結果",
            ""
        ])
        
        if validation["issues"]:
            report_lines.extend([
                "### ⚠️ 發現的問題",
                ""
            ])
            for issue in validation["issues"]:
                report_lines.append(f"- ⚠️ {issue}")
        else:
            report_lines.append("✅ **所有驗證通過，無發現問題**")
        
        # 添加後續建議
        report_lines.extend([
            "",
            "## 💡 後續建議",
            "",
            "1. **測試新結構**",
            "   ```bash",
            "   # 測試 Docker Compose",
            "   docker-compose -f docker/compose/docker-compose.yml config",
            "   ",
            "   # 測試 Kubernetes 配置",
            "   kubectl apply --dry-run=client -f docker/k8s/",
            "   ```",
            "",
            "2. **更新 CI/CD 管道**",
            "   - 更新構建腳本中的 Dockerfile 路徑",
            "   - 更新部署腳本中的 docker-compose 路徑",
            "",
            "3. **團隊通知**",
            "   - 通知開發團隊新的目錄結構",
            "   - 更新開發文檔和 README",
            "",
            "4. **清理工作**",
            f"   - 確認新結構正常運行後，可刪除備份: `{self.backup_dir}`",
            "",
            "## 📚 相關文檔",
            "",
            "- [Docker 使用指南](docker/DOCKER_GUIDE.md)",
            "- [Docker 基礎設施分析報告](docker_infrastructure_analysis_20251030_113318.md)",
            "- [aiva_common README](services/aiva_common/README.md)",
            "",
            f"---",
            f"*報告生成時間: {datetime.now().isoformat()}*"
        ])
        
        return "\n".join(report_lines)
    
    def save_update_report(self, output_file: Optional[str] = None) -> str:
        """保存更新報告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"docker_infrastructure_update_report_{timestamp}.md"
        
        output_path = self.repo_path / output_file
        
        # 生成報告內容
        report_content = self.generate_update_report()
        
        # 添加時間戳標頭
        timestamped_content = f"""---
Created: {datetime.now().strftime("%Y-%m-%d")}
Last Modified: {datetime.now().strftime("%Y-%m-%d")}
Document Type: Docker Infrastructure Update Report
---

{report_content}
"""
        
        # 保存報告
        output_path.write_text(timestamped_content, encoding='utf-8')
        
        return str(output_path)
    
    def run_full_update(self) -> str:
        """執行完整的 Docker 基礎設施更新"""
        print("🔧 開始 Docker 基礎設施更新...")
        
        try:
            # 1. 創建備份
            print("💾 創建備份...")
            self.create_backup()
            
            # 2. 分析當前結構
            print("🔍 分析當前結構...")
            analysis = self.analyze_current_docker_structure()
            print(f"   發現 {len(analysis['root_dockerfiles'])} 個 Dockerfile")
            print(f"   發現 {len(analysis['compose_files'])} 個 Compose 文件")
            
            # 3. 創建新目錄結構
            print("🏗️ 創建新目錄結構...")
            self.create_new_docker_structure()
            
            # 4. 重組 Docker 文件
            print("📁 重組 Docker 文件...")
            self.reorganize_docker_files()
            
            # 5. 更新文檔
            print("📚 更新相關文檔...")
            self.update_documentation()
            
            # 6. 驗證結果
            print("🔍 驗證新結構...")
            validation = self.validate_new_structure()
            
            if validation["issues"]:
                print(f"⚠️ 發現 {len(validation['issues'])} 個問題")
                for issue in validation["issues"]:
                    print(f"   - {issue}")
            else:
                print("✅ 所有驗證通過")
            
            # 7. 生成報告
            print("📄 生成更新報告...")
            report_path = self.save_update_report()
            
            print(f"✅ Docker 基礎設施更新完成！")
            print(f"   - 備份位置: {self.backup_dir}")
            print(f"   - 報告位置: {report_path}")
            
            return report_path
            
        except Exception as e:
            self.log_update("更新失敗", f"發生錯誤: {str(e)}", False)
            print(f"❌ 更新失敗: {e}")
            raise

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Docker 基礎設施更新工具")
    parser.add_argument("--repo-path", default=".", help="專案路徑 (預設: 當前目錄)")
    parser.add_argument("--dry-run", action="store_true", help="僅分析，不執行實際更新")
    parser.add_argument("--backup-only", action="store_true", help="僅創建備份")
    
    args = parser.parse_args()
    
    # 創建更新器
    updater = DockerInfrastructureUpdater(args.repo_path)
    
    if args.backup_only:
        # 只創建備份
        backup_files = updater.create_backup()
        print(f"✅ 備份完成: {len(backup_files)} 個文件")
        print(f"   備份位置: {updater.backup_dir}")
    elif args.dry_run:
        # 只分析不執行
        print("🔍 執行分析（乾燥運行）...")
        analysis = updater.analyze_current_docker_structure()
        
        print(f"\n📊 分析結果:")
        print(f"   - Dockerfile 數量: {len(analysis['root_dockerfiles'])}")
        print(f"   - Compose 文件數量: {len(analysis['compose_files'])}")
        print(f"   - K8s 文件數量: {len(analysis['k8s_files'])}")
        print(f"   - Helm 文件數量: {len(analysis['helm_files'])}")
        print(f"   - 服務總數: {analysis['service_mapping']['service_count']}")
        
        print(f"\n💡 建議操作:")
        print(f"   - 移動 {len(analysis['root_dockerfiles'])} 個 Dockerfile 到專門目錄")
        print(f"   - 重組 {len(analysis['compose_files'])} 個 Compose 配置")
        print(f"   - 整合 K8s 和 Helm 配置到統一結構")
        
        print(f"\n▶️ 執行實際更新: python {__file__} --repo-path {args.repo_path}")
    else:
        # 執行完整更新
        report_path = updater.run_full_update()
        print(f"\n📊 更新摘要:")
        print(f"   - 操作數量: {len(updater.update_log)}")
        print(f"   - 備份位置: {updater.backup_dir}")
        print(f"   - 報告位置: {report_path}")

if __name__ == "__main__":
    main()