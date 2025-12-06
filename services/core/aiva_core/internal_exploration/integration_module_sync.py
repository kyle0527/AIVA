"""Integration Module Sync - 整合模塊數據同步

將 internal_exploration 的分析結果同步到 AIVA 的統一數據存儲
包括 attack_graph.pkl 和 experience.db 的數據整合

作者: AIVA Team
版本: 1.0
日期: 2025-12-06
"""

import json
import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx

# 導入分析組件
try:
    from .path_difference_analyzer import PathDifferenceAnalyzer, PathAnalysisReport
    from .capability_classifier import CapabilityClassifier, CapabilityInfo
    from .capability_command_builder import CapabilityCommandBuilder, CommandSequence
except ImportError:
    from path_difference_analyzer import PathDifferenceAnalyzer, PathAnalysisReport
    from capability_classifier import CapabilityClassifier, CapabilityInfo
    from capability_command_builder import CapabilityCommandBuilder, CommandSequence

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """同步報告"""
    timestamp: str
    source_analysis_dir: str
    target_integration_dir: str
    flows_synced: int
    capabilities_synced: int
    commands_synced: int
    graph_nodes_added: int
    graph_edges_added: int
    db_records_added: int
    success: bool
    error_messages: List[str]


class IntegrationModuleSync:
    """整合模塊數據同步器
    
    核心功能：
    1. 將流程分析結果同步到 NetworkX 圖結構
    2. 將能力分類結果存儲到經驗數據庫
    3. 將指令序列保存為可執行腳本
    4. 維護數據一致性和版本控制
    5. 提供數據查詢和檢索接口
    """
    
    def __init__(self, 
                 project_root: Path, 
                 integration_dir: Path = None):
        """初始化同步器
        
        Args:
            project_root: 項目根目錄
            integration_dir: 整合模塊目錄，默認為 services/integration
        """
        self.project_root = Path(project_root)
        
        if integration_dir:
            self.integration_dir = Path(integration_dir)
        else:
            self.integration_dir = self.project_root / "services" / "integration"
            
        # 確保整合目錄存在
        self.integration_dir.mkdir(parents=True, exist_ok=True)
        
        # 關鍵數據文件路徑
        self.attack_graph_path = self.integration_dir / "attack_graph.pkl"
        self.experience_db_path = self.integration_dir / "experience.db" 
        self.capability_data_dir = self.integration_dir / "capability_data"
        self.command_scripts_dir = self.integration_dir / "executable_commands"
        
        # 確保子目錄存在
        self.capability_data_dir.mkdir(exist_ok=True)
        self.command_scripts_dir.mkdir(exist_ok=True)
        
        logger.info(f"初始化整合模塊同步器: {self.integration_dir}")
        
    def _load_attack_graph(self) -> nx.DiGraph:
        """載入攻擊圖，如果不存在則創建新圖"""
        if self.attack_graph_path.exists():
            try:
                with open(self.attack_graph_path, 'rb') as f:
                    graph = pickle.load(f)
                logger.info(f"載入現有攻擊圖: {len(graph.nodes)} 節點, {len(graph.edges)} 邊")
                return graph
            except Exception as e:
                logger.warning(f"載入攻擊圖失敗: {e}，創建新圖")
                
        # 創建新圖
        graph = nx.DiGraph()
        graph.graph['metadata'] = {
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'source': 'internal_exploration_sync'
        }
        return graph
        
    def _save_attack_graph(self, graph: nx.DiGraph) -> None:
        """保存攻擊圖"""
        try:
            # 更新元數據
            graph.graph['metadata']['last_updated'] = datetime.now().isoformat()
            graph.graph['metadata']['nodes_count'] = len(graph.nodes)
            graph.graph['metadata']['edges_count'] = len(graph.edges)
            
            with open(self.attack_graph_path, 'wb') as f:
                pickle.dump(graph, f)
            logger.info(f"攻擊圖已保存: {self.attack_graph_path}")
        except Exception as e:
            logger.error(f"保存攻擊圖失敗: {e}")
            raise
            
    def _init_experience_db(self) -> sqlite3.Connection:
        """初始化經驗數據庫"""
        conn = sqlite3.connect(str(self.experience_db_path))
        
        # 創建表結構
        conn.executescript("""
            -- 能力表
            CREATE TABLE IF NOT EXISTS capabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                frequency INTEGER DEFAULT 0,
                complexity_score REAL DEFAULT 0.0,
                risk_level TEXT DEFAULT 'low',
                dependencies TEXT, -- JSON 格式
                implementation_paths TEXT, -- JSON 格式
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- 流程路徑表  
            CREATE TABLE IF NOT EXISTS flow_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path_id TEXT UNIQUE NOT NULL,
                capability_name TEXT,
                nodes TEXT NOT NULL, -- JSON 格式
                entry_point TEXT,
                endpoint TEXT,
                frequency INTEGER DEFAULT 1,
                length INTEGER,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (capability_name) REFERENCES capabilities(name)
            );
            
            -- 指令序列表
            CREATE TABLE IF NOT EXISTS command_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id TEXT UNIQUE NOT NULL,
                capability_name TEXT,
                validation_status TEXT,
                total_steps INTEGER DEFAULT 0,
                estimated_time REAL DEFAULT 0.0,
                execution_count INTEGER DEFAULT 0,
                last_executed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (capability_name) REFERENCES capabilities(name)
            );
            
            -- 路徑分析報告表
            CREATE TABLE IF NOT EXISTS path_analysis_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                total_paths INTEGER,
                unique_paths INTEGER,
                recommended_path TEXT,
                analysis_summary TEXT,
                metrics_data TEXT, -- JSON 格式
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- 同步歷史表
            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_dir TEXT NOT NULL,
                flows_synced INTEGER DEFAULT 0,
                capabilities_synced INTEGER DEFAULT 0,
                commands_synced INTEGER DEFAULT 0,
                success BOOLEAN DEFAULT FALSE,
                error_message TEXT,
                synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- 創建索引
            CREATE INDEX IF NOT EXISTS idx_capabilities_category ON capabilities(category);
            CREATE INDEX IF NOT EXISTS idx_flow_paths_endpoint ON flow_paths(endpoint);
            CREATE INDEX IF NOT EXISTS idx_command_sequences_capability ON command_sequences(capability_name);
        """)
        
        conn.commit()
        logger.info(f"經驗數據庫已初始化: {self.experience_db_path}")
        return conn
        
    def sync_flow_analysis_to_graph(self, 
                                   flow_analysis_dir: Path,
                                   analyzer: PathDifferenceAnalyzer = None) -> Tuple[int, int]:
        """將流程分析結果同步到攻擊圖
        
        Returns:
            (nodes_added, edges_added): 新增節點數和邊數
        """
        if not analyzer:
            analyzer = PathDifferenceAnalyzer(flow_analysis_dir)
            
        graph = self._load_attack_graph()
        nodes_before = len(graph.nodes)
        edges_before = len(graph.edges)
        
        # 為每個流程路徑添加節點和邊
        for flow in analyzer.flows:
            # 添加節點
            for i, node in enumerate(flow.nodes):
                if not graph.has_node(node):
                    graph.add_node(node, 
                        type='capability_endpoint' if i == len(flow.nodes) - 1 else 'intermediate_node',
                        first_seen=datetime.now().isoformat(),
                        frequency=0,
                        source='internal_exploration'
                    )
                    
                # 更新頻率
                graph.nodes[node]['frequency'] = graph.nodes[node].get('frequency', 0) + 1
                
            # 添加邊
            for i in range(len(flow.nodes) - 1):
                source = flow.nodes[i]
                target = flow.nodes[i + 1]
                
                if graph.has_edge(source, target):
                    # 更新現有邊的權重
                    graph.edges[source, target]['weight'] += 1
                else:
                    # 添加新邊
                    graph.add_edge(source, target, 
                        weight=1,
                        flow_type='data_flow',
                        source='internal_exploration',
                        first_seen=datetime.now().isoformat()
                    )
                    
        # 添加圖級別的統計信息
        graph.graph['analysis_stats'] = {
            'total_flows': len(analyzer.flows),
            'unique_endpoints': len(analyzer.endpoint_groups),
            'last_sync': datetime.now().isoformat(),
            'source_dir': str(flow_analysis_dir)
        }
        
        self._save_attack_graph(graph)
        
        nodes_added = len(graph.nodes) - nodes_before
        edges_added = len(graph.edges) - edges_before
        
        logger.info(f"流程同步完成: +{nodes_added} 節點, +{edges_added} 邊")
        return nodes_added, edges_added
        
    def sync_capabilities_to_db(self, classifier: CapabilityClassifier) -> int:
        """將能力分類結果同步到經驗數據庫
        
        Returns:
            synced_count: 同步的能力數量
        """
        conn = self._init_experience_db()
        synced_count = 0
        
        try:
            for name, capability in classifier.capabilities.items():
                # 檢查是否已存在
                cursor = conn.execute(
                    "SELECT id FROM capabilities WHERE name = ?",
                    (name,)
                )
                
                if cursor.fetchone():
                    # 更新現有記錄
                    conn.execute("""
                        UPDATE capabilities SET
                            category = ?, description = ?, frequency = ?,
                            complexity_score = ?, risk_level = ?,
                            dependencies = ?, implementation_paths = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE name = ?
                    """, (
                        capability.category.value,
                        capability.description,
                        capability.frequency,
                        capability.complexity_score,
                        capability.risk_level,
                        json.dumps(list(capability.dependencies)),
                        json.dumps(capability.implementation_paths),
                        name
                    ))
                else:
                    # 插入新記錄
                    conn.execute("""
                        INSERT INTO capabilities 
                        (name, category, description, frequency, complexity_score, 
                         risk_level, dependencies, implementation_paths)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        capability.category.value,
                        capability.description,
                        capability.frequency,
                        capability.complexity_score,
                        capability.risk_level,
                        json.dumps(list(capability.dependencies)),
                        json.dumps(capability.implementation_paths)
                    ))
                    
                synced_count += 1
                
            conn.commit()
            logger.info(f"能力同步完成: {synced_count} 個能力")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"能力同步失敗: {e}")
            raise
        finally:
            conn.close()
            
        return synced_count
        
    def sync_flow_paths_to_db(self, analyzer: PathDifferenceAnalyzer) -> int:
        """將流程路徑同步到數據庫
        
        Returns:
            synced_count: 同步的路徑數量
        """
        conn = self._init_experience_db()
        synced_count = 0
        
        try:
            for flow in analyzer.flows:
                # 檢查是否已存在
                cursor = conn.execute(
                    "SELECT id FROM flow_paths WHERE path_id = ?",
                    (flow.id,)
                )
                
                if not cursor.fetchone():
                    # 插入新路徑
                    conn.execute("""
                        INSERT INTO flow_paths 
                        (path_id, capability_name, nodes, entry_point, 
                         endpoint, frequency, length, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        flow.id,
                        flow.endpoint,  # 使用終點作為能力名稱
                        json.dumps(flow.nodes),
                        flow.entry_point,
                        flow.endpoint,
                        flow.frequency,
                        flow.length,
                        flow.source_file
                    ))
                    synced_count += 1
                    
            conn.commit()
            logger.info(f"流程路徑同步完成: {synced_count} 個路徑")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"流程路徑同步失敗: {e}")
            raise
        finally:
            conn.close()
            
        return synced_count
        
    def sync_command_sequences_to_db(self, builder: CapabilityCommandBuilder) -> int:
        """將指令序列同步到數據庫
        
        Returns:
            synced_count: 同步的序列數量
        """
        conn = self._init_experience_db()
        synced_count = 0
        
        try:
            for sequence_id, sequence in builder.command_sequences.items():
                # 檢查是否已存在
                cursor = conn.execute(
                    "SELECT id FROM command_sequences WHERE sequence_id = ?",
                    (sequence_id,)
                )
                
                if cursor.fetchone():
                    # 更新執行計數
                    conn.execute("""
                        UPDATE command_sequences SET
                            execution_count = ?,
                            last_executed_at = CURRENT_TIMESTAMP
                        WHERE sequence_id = ?
                    """, (
                        len(sequence.execution_history),
                        sequence_id
                    ))
                else:
                    # 插入新序列
                    conn.execute("""
                        INSERT INTO command_sequences 
                        (sequence_id, capability_name, validation_status,
                         total_steps, estimated_time, execution_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        sequence_id,
                        sequence.capability_name,
                        sequence.validation_status,
                        len(sequence.steps),
                        sequence.total_estimated_time,
                        len(sequence.execution_history)
                    ))
                    synced_count += 1
                    
                # 保存詳細的指令序列數據為 JSON
                sequence_data_file = self.capability_data_dir / f"{sequence_id}.json"
                with open(sequence_data_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'sequence_id': sequence_id,
                        'capability_name': sequence.capability_name,
                        'validation_status': sequence.validation_status,
                        'steps': [
                            {
                                'script_name': step.script_name,
                                'command': step.command,
                                'is_executable': step.is_executable,
                                'parameters': step.parameters,
                                'execution_time': step.execution_time
                            }
                            for step in sequence.steps
                        ],
                        'validation_messages': sequence.validation_messages,
                        'execution_history': sequence.execution_history
                    }, f, indent=2, ensure_ascii=False)
                    
            conn.commit()
            logger.info(f"指令序列同步完成: {synced_count} 個序列")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"指令序列同步失敗: {e}")
            raise
        finally:
            conn.close()
            
        return synced_count
        
    def sync_path_analysis_reports_to_db(self, 
                                        reports: Dict[str, PathAnalysisReport]) -> int:
        """將路徑分析報告同步到數據庫
        
        Returns:
            synced_count: 同步的報告數量
        """
        conn = self._init_experience_db()
        synced_count = 0
        
        try:
            for endpoint, report in reports.items():
                # 刪除舊報告
                conn.execute("DELETE FROM path_analysis_reports WHERE endpoint = ?", (endpoint,))
                
                # 插入新報告
                conn.execute("""
                    INSERT INTO path_analysis_reports 
                    (endpoint, total_paths, unique_paths, recommended_path,
                     analysis_summary, metrics_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    endpoint,
                    report.total_paths,
                    report.unique_paths,
                    report.recommended_path,
                    report.analysis_summary,
                    json.dumps({
                        pattern: {
                            'efficiency': metrics.efficiency_score,
                            'risk': metrics.risk_score,
                            'cost': metrics.cost_score,
                            'popularity': metrics.popularity_score,
                            'overall': metrics.overall_score
                        }
                        for pattern, metrics in report.path_metrics.items()
                    })
                ))
                synced_count += 1
                
            conn.commit()
            logger.info(f"路徑分析報告同步完成: {synced_count} 個報告")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"路徑分析報告同步失敗: {e}")
            raise
        finally:
            conn.close()
            
        return synced_count
        
    def full_sync(self, flow_analysis_dir: Path) -> SyncReport:
        """執行完整同步
        
        Args:
            flow_analysis_dir: 流程分析結果目錄
            
        Returns:
            SyncReport: 完整的同步報告
        """
        start_time = datetime.now()
        error_messages = []
        
        logger.info(f"開始完整同步: {flow_analysis_dir}")
        
        try:
            # 1. 初始化各個分析器
            analyzer = PathDifferenceAnalyzer(flow_analysis_dir)
            classifier = CapabilityClassifier()
            classifier.load_capabilities_from_flow_analysis(flow_analysis_dir)
            builder = CapabilityCommandBuilder(self.project_root)
            
            # 2. 同步到攻擊圖
            nodes_added, edges_added = self.sync_flow_analysis_to_graph(
                flow_analysis_dir, analyzer
            )
            
            # 3. 同步能力到數據庫
            capabilities_synced = self.sync_capabilities_to_db(classifier)
            
            # 4. 同步流程路徑到數據庫
            flows_synced = self.sync_flow_paths_to_db(analyzer)
            
            # 5. 生成路徑分析報告並同步
            multi_path_reports = analyzer.analyze_all_multi_path_endpoints()
            reports_synced = self.sync_path_analysis_reports_to_db(multi_path_reports)
            
            # 6. 為主要能力構建指令序列
            commands_synced = 0
            for endpoint in ['models', 'orchestrator', 'core', 'store'][:5]:  # 限制數量
                if endpoint in analyzer.endpoint_groups:
                    paths = analyzer.endpoint_groups[endpoint]
                    if paths:
                        first_path = paths[0]
                        try:
                            builder.build_command_sequence(endpoint, first_path.nodes)
                            commands_synced += 1
                        except Exception as e:
                            error_messages.append(f"構建 {endpoint} 指令失敗: {e}")
                            
            # 7. 同步指令序列到數據庫
            sequences_synced = self.sync_command_sequences_to_db(builder)
            
            # 8. 記錄同步歷史
            conn = self._init_experience_db()
            try:
                conn.execute("""
                    INSERT INTO sync_history 
                    (source_dir, flows_synced, capabilities_synced, 
                     commands_synced, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(flow_analysis_dir),
                    flows_synced,
                    capabilities_synced,
                    commands_synced,
                    len(error_messages) == 0,
                    "; ".join(error_messages) if error_messages else None
                ))
                conn.commit()
            finally:
                conn.close()
                
            # 創建同步報告
            report = SyncReport(
                timestamp=start_time.isoformat(),
                source_analysis_dir=str(flow_analysis_dir),
                target_integration_dir=str(self.integration_dir),
                flows_synced=flows_synced,
                capabilities_synced=capabilities_synced,
                commands_synced=commands_synced,
                graph_nodes_added=nodes_added,
                graph_edges_added=edges_added,
                db_records_added=flows_synced + capabilities_synced + commands_synced + reports_synced,
                success=len(error_messages) == 0,
                error_messages=error_messages
            )
            
            # 保存同步報告
            report_file = self.integration_dir / f"sync_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
                
            logger.info(f"完整同步完成: {report.success}")
            return report
            
        except Exception as e:
            error_messages.append(f"同步過程異常: {e}")
            logger.error(f"完整同步失敗: {e}")
            
            return SyncReport(
                timestamp=start_time.isoformat(),
                source_analysis_dir=str(flow_analysis_dir),
                target_integration_dir=str(self.integration_dir),
                flows_synced=0,
                capabilities_synced=0,
                commands_synced=0,
                graph_nodes_added=0,
                graph_edges_added=0,
                db_records_added=0,
                success=False,
                error_messages=error_messages
            )
            
    def query_capabilities(self, 
                          category: str = None,
                          min_frequency: int = 0) -> List[Dict[str, Any]]:
        """查詢能力數據
        
        Args:
            category: 能力分類過濾
            min_frequency: 最小使用頻率
            
        Returns:
            List[Dict]: 能力列表
        """
        conn = self._init_experience_db()
        
        query = "SELECT * FROM capabilities WHERE frequency >= ?"
        params = [min_frequency]
        
        if category:
            query += " AND category = ?"
            params.append(category)
            
        query += " ORDER BY frequency DESC"
        
        try:
            cursor = conn.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    'name': row[1],
                    'category': row[2],
                    'description': row[3],
                    'frequency': row[4],
                    'complexity_score': row[5],
                    'risk_level': row[6],
                    'dependencies': json.loads(row[7]) if row[7] else [],
                    'implementation_paths': json.loads(row[8]) if row[8] else []
                })
                
            return results
            
        finally:
            conn.close()
            
    def get_attack_graph_stats(self) -> Dict[str, Any]:
        """獲取攻擊圖統計信息"""
        if not self.attack_graph_path.exists():
            return {'error': '攻擊圖不存在'}
            
        try:
            graph = self._load_attack_graph()
            
            # 計算圖統計信息
            stats = {
                'nodes_count': len(graph.nodes),
                'edges_count': len(graph.edges),
                'connected_components': nx.number_connected_components(graph.to_undirected()),
                'density': nx.density(graph),
                'metadata': graph.graph.get('metadata', {}),
                'analysis_stats': graph.graph.get('analysis_stats', {}),
                'top_nodes_by_frequency': []
            }
            
            # 獲取頻率最高的節點
            node_frequencies = [(node, data.get('frequency', 0)) 
                              for node, data in graph.nodes(data=True)]
            top_nodes = sorted(node_frequencies, key=lambda x: x[1], reverse=True)[:10]
            stats['top_nodes_by_frequency'] = top_nodes
            
            return stats
            
        except Exception as e:
            return {'error': f'獲取攻擊圖統計失敗: {e}'}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="整合模塊數據同步器")
    parser.add_argument("--project-root", default=".", help="項目根目錄")
    parser.add_argument("--flow-dir", required=True, help="流程分析結果目錄")
    parser.add_argument("--integration-dir", help="整合模塊目錄")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    # 執行同步
    syncer = IntegrationModuleSync(
        Path(args.project_root),
        Path(args.integration_dir) if args.integration_dir else None
    )
    
    report = syncer.full_sync(Path(args.flow_dir))
    
    print("=== 整合模塊同步報告 ===")
    print(f"同步時間: {report.timestamp}")
    print(f"同步結果: {'✅ 成功' if report.success else '❌ 失敗'}")
    print(f"流程同步: {report.flows_synced}")
    print(f"能力同步: {report.capabilities_synced}")
    print(f"指令同步: {report.commands_synced}")
    print(f"圖節點: +{report.graph_nodes_added}")
    print(f"圖邊: +{report.graph_edges_added}")
    print(f"數據庫記錄: +{report.db_records_added}")
    
    if report.error_messages:
        print("\n錯誤訊息:")
        for msg in report.error_messages:
            print(f"  ❌ {msg}")
            
    print(f"\n同步完成，數據已保存至: {syncer.integration_dir}")