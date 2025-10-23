"""
經驗資料庫 (Experience Replay Memory) 模型定義

集中存放每次攻擊執行結果的資料表，包括：
- 環境與場景資訊
- AST 攻擊計畫
- 實際執行 Trace
- 差異指標/回饋
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ExperienceRecord(Base):
    """經驗記錄

    存儲單次攻擊執行的完整經驗
    """

    __tablename__ = "experience_records"

    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    experience_id = Column(String(100), unique=True, nullable=False, index=True)

    # 時間戳記
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    # 執行計畫資訊
    plan_id = Column(String(100), nullable=False, index=True)
    attack_type = Column(String(50), nullable=False, index=True)
    target_info = Column(JSON, nullable=True)  # 目標資訊

    # AST 攻擊計畫 (JSON 格式)
    ast_graph = Column(JSON, nullable=False)  # 完整的 AST 圖結構

    # 執行 Trace (JSON 格式)
    trace_session_id = Column(String(100), nullable=False, index=True)
    execution_trace = Column(JSON, nullable=False)  # 完整的執行軌跡

    # 比較指標
    completion_rate = Column(Float, nullable=True)  # 完成率
    sequence_match_rate = Column(Float, nullable=True)  # 順序匹配率
    overall_score = Column(Float, nullable=True, index=True)  # 綜合評分
    metrics_detail = Column(JSON, nullable=True)  # 詳細指標

    # 回饋信號
    reward = Column(Float, nullable=True)  # 強化學習獎勵值
    feedback_data = Column(JSON, nullable=True)  # 完整回饋數據

    # 執行結果
    execution_success = Column(Integer, nullable=True)  # 成功步驟數
    execution_failed = Column(Integer, nullable=True)  # 失敗步驟數
    error_count = Column(Integer, nullable=True)  # 錯誤數量

    # 額外元數據
    extra_metadata = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)  # 備註

    def __repr__(self) -> str:
        return (
            f"<ExperienceRecord(id={self.id}, "
            f"experience_id='{self.experience_id}', "
            f"attack_type='{self.attack_type}', "
            f"score={self.overall_score})>"
        )


class TrainingDataset(Base):
    """訓練資料集

    用於組織和標記訓練樣本
    """

    __tablename__ = "training_datasets"

    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(100), unique=True, nullable=False, index=True)

    # 基本資訊
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    # 資料集配置
    attack_types = Column(JSON, nullable=True)  # 包含的攻擊類型
    min_score_threshold = Column(Float, nullable=True)  # 最低分數閾值
    max_samples = Column(Integer, nullable=True)  # 最大樣本數

    # 樣本選擇標準
    selection_criteria = Column(JSON, nullable=True)

    # 統計資訊
    total_samples = Column(Integer, default=0)
    avg_score = Column(Float, nullable=True)

    # 元數據
    extra_metadata = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<TrainingDataset(id={self.id}, "
            f"name='{self.name}', "
            f"samples={self.total_samples})>"
        )


class DatasetSample(Base):
    """資料集樣本關聯

    連接經驗記錄與訓練資料集
    """

    __tablename__ = "dataset_samples"

    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 外鍵
    dataset_id = Column(String(100), nullable=False, index=True)
    experience_id = Column(String(100), nullable=False, index=True)

    # 樣本資訊
    added_at = Column(DateTime, default=datetime.now, nullable=False)
    sample_weight = Column(Float, default=1.0)  # 樣本權重
    is_validated = Column(Integer, default=0)  # 是否已驗證

    # 標註資訊
    label = Column(String(50), nullable=True)  # 標籤 (positive/negative/neutral)
    annotator = Column(String(100), nullable=True)  # 標註者
    annotation_notes = Column(Text, nullable=True)

    # 元數據
    extra_metadata = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DatasetSample(dataset={self.dataset_id}, "
            f"experience={self.experience_id}, "
            f"label={self.label})>"
        )


class ModelTrainingHistory(Base):
    """模型訓練歷史

    記錄每次模型訓練的詳細信息
    """

    __tablename__ = "model_training_history"

    # 主鍵
    id = Column(Integer, primary_key=True, autoincrement=True)
    training_id = Column(String(100), unique=True, nullable=False, index=True)

    # 訓練基本資訊
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # 訓練配置
    dataset_id = Column(String(100), nullable=False)
    training_config = Column(JSON, nullable=True)  # 訓練參數
    hyperparameters = Column(JSON, nullable=True)  # 超參數

    # 訓練結果
    status = Column(String(50), nullable=False)  # running/completed/failed
    final_loss = Column(Float, nullable=True)
    final_accuracy = Column(Float, nullable=True)
    training_metrics = Column(JSON, nullable=True)  # 完整訓練指標

    # 模型資訊
    model_path = Column(String(500), nullable=True)  # 模型儲存路徑
    model_size_mb = Column(Float, nullable=True)

    # 評估結果
    validation_metrics = Column(JSON, nullable=True)
    test_metrics = Column(JSON, nullable=True)

    # 元數據
    extra_metadata = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ModelTrainingHistory(id={self.id}, "
            f"model='{self.model_name}', "
            f"version='{self.model_version}', "
            f"status='{self.status}')>"
        )
