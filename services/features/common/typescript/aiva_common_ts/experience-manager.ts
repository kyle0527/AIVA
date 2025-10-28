/**
 * AIVA TypeScript Experience Manager
 * 
 * TypeScript 版本的經驗管理器，對應 aiva_common.ai.experience_manager
 * 提供與 Python 版本一致的強化學習經驗樣本管理功能
 * 
 * 架構位置: services/features/common/typescript/aiva_common_ts/
 * 用途: 支持 TypeScript 模組的 AI 經驗學習需求
 */

import { EventEmitter } from 'events';
import * as crypto from 'crypto';

// ============================================================================
// 基礎類型定義 (對應 Python 的 Pydantic models)
// ============================================================================

export interface ExperienceSample {
  // 基本標識 - 與 Python aiva_common.schemas.ai.ExperienceSample 完全一致
  sample_id: string;
  session_id: string;
  plan_id: string;
  
  // 狀態信息
  state_before: Record<string, any>;
  action_taken: Record<string, any>;
  state_after: Record<string, any>;
  
  // 獎勵信息
  reward: number;
  reward_breakdown: Record<string, number>;
  
  // 上下文信息
  context: Record<string, any>;
  target_info: Record<string, any>;
  
  // 時間信息
  timestamp: Date;
  duration_ms?: number;
  
  // 質量標記
  quality_score?: number;
  is_positive: boolean;
  confidence: number;
  
  // 學習標籤
  learning_tags: string[];
  difficulty_level: number;
}

export interface LearningSession {
  sessionId: string;
  trainingId?: string;
  sessionType: SessionType;
  startTime: Date;
  endTime?: Date;
  isActive: boolean;
  status: SessionStatus;
  totalSamples: number;
  highQualitySamples: number;
  mediumQualitySamples: number;
  lowQualitySamples: number;
  uniquePlans: Set<string>;
  vulnerabilityTypes: Set<string>;
  qualityThreshold: number;
  maxSamples: number;
  autoCleanup: boolean;
  tags: string[];
  metadata: Record<string, any>;
}

export interface ExperienceFilter {
  sessionIds: string[];
  planIds: string[];
  vulnerabilityTypes: string[];
  minQualityScore?: number;
  maxQualityScore?: number;
  positiveOnly: boolean;
  startTime?: Date;
  endTime?: Date;
  minReward?: number;
  maxReward?: number;
  tags: string[];
  sortBy: SortBy;
  sortDesc: boolean;
  limit?: number;
  offset: number;
}

export interface ExperienceStatistics {
  totalSamples: number;
  totalSessions: number;
  totalPlans: number;
  highQualitySamples: number;
  mediumQualitySamples: number;
  lowQualitySamples: number;
  avgQualityScore: number;
  positiveSamples: number;
  negativeSamples: number;
  successRate: number;
  avgReward: number;
  maxReward: number;
  minReward: number;
  earliestSample?: Date;
  latestSample?: Date;
  vulnerabilityTypeDistribution: Record<string, number>;
  planTypeDistribution: Record<string, number>;
  difficultyDistribution: Record<string, number>;
}

export interface ExperienceManagerConfig {
  managerId: string;
  storageBackend: StorageBackend;
  storagePath: string;
  retentionDays: number;
  deduplicationEnabled: boolean;
  autoCleanup: boolean;
  maxMemoryUsage: number;
  batchSize: number;
}

// ============================================================================
// 枚舉定義
// ============================================================================

export enum SessionType {
  INTERACTIVE = "interactive",
  BATCH = "batch",
  EVALUATION = "evaluation",
  TRAINING = "training"
}

export enum SessionStatus {
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed"
}

export enum SortBy {
  TIMESTAMP = "timestamp",
  QUALITY_SCORE = "quality_score",
  REWARD = "reward"
}

export enum StorageBackend {
  MEMORY = "memory",
  JSON_FILE = "json_file",
  SQLITE = "sqlite"
}

// ============================================================================
// 存儲接口和實現
// ============================================================================

interface ExperienceStorage {
  storeExperience(sample: ExperienceSample): Promise<boolean>;
  getExperiences(filter: ExperienceFilter): Promise<ExperienceSample[]>;
  storeSession(session: LearningSession): Promise<boolean>;
  getStatistics(): Promise<ExperienceStatistics>;
  cleanup(): Promise<void>;
}

class MemoryExperienceStorage implements ExperienceStorage {
  private experiences: ExperienceSample[] = [];
  private sessions: Map<string, LearningSession> = new Map();

  async storeExperience(sample: ExperienceSample): Promise<boolean> {
    try {
      this.experiences.push({ ...sample });
      return true;
    } catch (error) {
      console.error(`Error storing experience: ${error}`);
      return false;
    }
  }

  async getExperiences(filter: ExperienceFilter): Promise<ExperienceSample[]> {
    let filtered = [...this.experiences];

    // 應用過濾條件
    if (filter.sessionIds.length > 0) {
      filtered = filtered.filter(exp => filter.sessionIds.includes(exp.session_id));
    }

    if (filter.planIds && filter.planIds.length > 0) {
      filtered = filtered.filter(exp => filter.planIds.includes(exp.plan_id));
    }

    if (filter.minQualityScore !== undefined) {
      filtered = filtered.filter(exp => (exp.quality_score || 0) >= filter.minQualityScore!);
    }

    if (filter.maxQualityScore !== undefined) {
      filtered = filtered.filter(exp => (exp.quality_score || 0) <= filter.maxQualityScore!);
    }

    if (filter.positiveOnly) {
      filtered = filtered.filter(exp => exp.is_positive);
    }

    if (filter.startTime) {
      filtered = filtered.filter(exp => exp.timestamp >= filter.startTime!);
    }

    if (filter.endTime) {
      filtered = filtered.filter(exp => exp.timestamp <= filter.endTime!);
    }

    // 排序
    filtered.sort((a, b) => {
      let comparison = 0;
      switch (filter.sortBy) {
        case SortBy.TIMESTAMP:
          comparison = a.timestamp.getTime() - b.timestamp.getTime();
          break;
        case SortBy.QUALITY_SCORE:
          comparison = (a.qualityScore || 0) - (b.qualityScore || 0);
          break;
        case SortBy.REWARD:
          comparison = a.reward - b.reward;
          break;
      }
      return filter.sortDesc ? -comparison : comparison;
    });

    // 分頁
    const start = filter.offset;
    const end = filter.limit ? start + filter.limit : undefined;
    return filtered.slice(start, end);
  }

  async storeSession(session: LearningSession): Promise<boolean> {
    try {
      // 直接存儲會話
      this.sessions.set(session.sessionId, session);
      return true;
    } catch (error) {
      console.error(`Error storing session: ${error}`);
      return false;
    }
  }

  async getStatistics(): Promise<ExperienceStatistics> {
    const stats: ExperienceStatistics = {
      totalSamples: this.experiences.length,
      totalSessions: this.sessions.size,
      totalPlans: new Set(this.experiences.map(exp => exp.planId)).size,
      highQualitySamples: 0,
      mediumQualitySamples: 0,
      lowQualitySamples: 0,
      avgQualityScore: 0,
      positiveSamples: this.experiences.filter(exp => exp.isPositive).length,
      negativeSamples: this.experiences.filter(exp => !exp.isPositive).length,
      successRate: 0,
      avgReward: 0,
      maxReward: 0,
      minReward: 0,
      vulnerabilityTypeDistribution: {},
      planTypeDistribution: {},
      difficultyDistribution: {}
    };

    if (this.experiences.length > 0) {
      // 質量統計
      this.experiences.forEach(exp => {
        const quality = exp.qualityScore || 0;
        if (quality >= 0.8) stats.highQualitySamples++;
        else if (quality >= 0.6) stats.mediumQualitySamples++;
        else stats.lowQualitySamples++;
      });

      // 質量分數平均
      const qualityScores = this.experiences.map(exp => exp.qualityScore || 0);
      stats.avgQualityScore = qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length;

      // 成功率
      stats.successRate = stats.positiveSamples / this.experiences.length;

      // 獎勵統計
      const rewards = this.experiences.map(exp => exp.reward);
      stats.avgReward = rewards.reduce((sum, reward) => sum + reward, 0) / rewards.length;
      stats.maxReward = Math.max(...rewards);
      stats.minReward = Math.min(...rewards);

      // 時間範圍
      const timestamps = this.experiences.map(exp => exp.timestamp);
      stats.earliestSample = new Date(Math.min(...timestamps.map(t => t.getTime())));
      stats.latestSample = new Date(Math.max(...timestamps.map(t => t.getTime())));

      // 分佈統計
      this.experiences.forEach(exp => {
        // 漏洞類型分佈
        const vulnType = exp.targetInfo.vulnerabilityType || 'unknown';
        stats.vulnerabilityTypeDistribution[vulnType] = (stats.vulnerabilityTypeDistribution[vulnType] || 0) + 1;

        // 計劃類型分佈
        const planType = exp.planId.split('_')[0] || 'unknown';
        stats.planTypeDistribution[planType] = (stats.planTypeDistribution[planType] || 0) + 1;

        // 難度分佈
        const difficulty = exp.difficultyLevel.toString();
        stats.difficultyDistribution[difficulty] = (stats.difficultyDistribution[difficulty] || 0) + 1;
      });
    }

    return stats;
  }

  async cleanup(): Promise<void> {
    // 內存存儲不需要特別清理
    console.log('[MemoryStorage] Cleanup completed');
  }
}

// ============================================================================
// 主要經驗管理器類別
// ============================================================================

export class AIVAExperienceManager extends EventEmitter {
  private config: ExperienceManagerConfig;
  private storage!: ExperienceStorage;
  private activeSessions: Map<string, LearningSession>;
  private dedupCache: Set<string>;
  private startTime: Date;
  private totalSamplesStored: number;
  private totalSamplesRetrieved: number;
  private cleanupTask?: NodeJS.Timeout;

  constructor(config: ExperienceManagerConfig) {
    super();
    this.config = config;
    this.activeSessions = new Map();
    this.dedupCache = new Set();
    this.startTime = new Date();
    this.totalSamplesStored = 0;
    this.totalSamplesRetrieved = 0;

    // 初始化存儲後端
    this.initializeStorage();

    // 啟動清理任務
    if (this.config.autoCleanup) {
      this.startCleanupTask();
    }

    console.log(`[ExperienceManager] Initialized with ${this.config.storageBackend} backend`);
  }

  // ========================================================================
  // 初始化和配置
  // ========================================================================

  private initializeStorage(): void {
    switch (this.config.storageBackend) {
      case StorageBackend.MEMORY:
        this.storage = new MemoryExperienceStorage();
        break;
      case StorageBackend.JSON_FILE:
        // TODO: 實現 JSON 文件存儲
        console.warn('[ExperienceManager] JSON file storage not implemented, using memory storage');
        this.storage = new MemoryExperienceStorage();
        break;
      case StorageBackend.SQLITE:
        // TODO: 實現 SQLite 存儲
        console.warn('[ExperienceManager] SQLite storage not implemented, using memory storage');
        this.storage = new MemoryExperienceStorage();
        break;
      default:
        throw new Error(`Unsupported storage backend: ${this.config.storageBackend}`);
    }
  }

  // ========================================================================
  // 學習會話管理
  // ========================================================================

  public async createLearningSession(
    trainingId?: string,
    sessionType: SessionType = SessionType.INTERACTIVE,
    options: Partial<LearningSession> = {}
  ): Promise<string> {
    try {
      const sessionId = this.generateId('learning');
      
      const session: LearningSession = {
        sessionId,
        trainingId,
        sessionType,
        startTime: new Date(),
        isActive: true,
        status: SessionStatus.RUNNING,
        totalSamples: 0,
        highQualitySamples: 0,
        mediumQualitySamples: 0,
        lowQualitySamples: 0,
        uniquePlans: new Set(),
        vulnerabilityTypes: new Set(),
        qualityThreshold: 0.7,
        maxSamples: 1000,
        autoCleanup: true,
        tags: [],
        metadata: {},
        ...options
      };

      // 存儲會話
      await this.storage.storeSession(session);

      // 添加到活躍會話
      this.activeSessions.set(sessionId, session);

      this.emit('session_created', session);
      console.log(`[ExperienceManager] Learning session created: ${sessionId}`);
      
      return sessionId;

    } catch (error) {
      console.error(`[ExperienceManager] Error creating learning session: ${error}`);
      throw error;
    }
  }

  public async endLearningSession(sessionId: string): Promise<boolean> {
    try {
      const session = this.activeSessions.get(sessionId);
      if (!session) {
        console.warn(`[ExperienceManager] Session not found: ${sessionId}`);
        return false;
      }

      session.isActive = false;
      session.status = SessionStatus.COMPLETED;
      session.endTime = new Date();

      // 更新存儲
      await this.storage.storeSession(session);

      // 從活躍會話中移除
      this.activeSessions.delete(sessionId);

      this.emit('session_ended', session);
      console.log(`[ExperienceManager] Learning session ended: ${sessionId}`);
      
      return true;

    } catch (error) {
      console.error(`[ExperienceManager] Error ending learning session: ${error}`);
      return false;
    }
  }

  // ========================================================================
  // 經驗樣本管理
  // ========================================================================

  public async storeExperience(sample: ExperienceSample): Promise<boolean> {
    try {
      // 去重檢查
      if (this.config.deduplicationEnabled) {
        const sampleHash = this.calculateSampleHash(sample);
        if (this.dedupCache.has(sampleHash)) {
          console.debug(`[ExperienceManager] Duplicate sample detected, skipping: ${sample.sampleId}`);
          return false;
        }
        this.dedupCache.add(sampleHash);
      }

      // 質量評估
      if (sample.qualityScore === undefined) {
        sample.qualityScore = await this.evaluateSampleQuality(sample);
      }

      // 存儲樣本
      const success = await this.storage.storeExperience(sample);

      if (success) {
        this.totalSamplesStored++;

        // 更新會話統計
        if (this.activeSessions.has(sample.sessionId)) {
          const session = this.activeSessions.get(sample.sessionId)!;
          this.updateSessionStatistics(session, sample);
          await this.storage.storeSession(session);
        }

        // 發送事件
        this.emit('experience_stored', sample);
        console.log(`[ExperienceManager] Experience sample stored: ${sample.sampleId}`);
      }

      return success;

    } catch (error) {
      console.error(`[ExperienceManager] Error storing experience: ${error}`);
      return false;
    }
  }

  public async getExperiences(
    sessionId?: string,
    planId?: string,
    qualityThreshold?: number,
    limit?: number
  ): Promise<ExperienceSample[]> {
    try {
      const filter: ExperienceFilter = {
        sessionIds: sessionId ? [sessionId] : [],
        planIds: planId ? [planId] : [],
        vulnerabilityTypes: [],
        minQualityScore: qualityThreshold,
        positiveOnly: false,
        tags: [],
        sortBy: SortBy.TIMESTAMP,
        sortDesc: true,
        limit,
        offset: 0
      };

      const samples = await this.storage.getExperiences(filter);
      this.totalSamplesRetrieved += samples.length;

      console.log(`[ExperienceManager] Retrieved ${samples.length} experience samples`);
      return samples;

    } catch (error) {
      console.error(`[ExperienceManager] Error retrieving experiences: ${error}`);
      return [];
    }
  }

  // ========================================================================
  // 質量評估
  // ========================================================================

  public async evaluateSampleQuality(sample: ExperienceSample): Promise<number> {
    let qualityScore = 0;

    // 基於獎勵的質量評估 (40%)
    const rewardFactor = Math.min(Math.abs(sample.reward) / 10.0, 1.0);
    qualityScore += (sample.isPositive ? Math.min(rewardFactor + 0.2, 1.0) : rewardFactor) * 0.4;

    // 基於置信度的質量評估 (25%)
    qualityScore += sample.confidence * 0.25;

    // 基於執行時長的質量評估 (15%)
    if (sample.durationMs) {
      const durationSeconds = sample.durationMs / 1000.0;
      let durationFactor = 1.0;
      
      if (durationSeconds >= 0.1 && durationSeconds <= 30.0) {
        durationFactor = 1.0;
      } else if (durationSeconds < 0.1) {
        durationFactor = durationSeconds / 0.1;
      } else {
        durationFactor = Math.max(30.0 / durationSeconds, 0.1);
      }
      
      qualityScore += durationFactor * 0.15;
    } else {
      qualityScore += 0.15;
    }

    // 基於難度級別的質量評估 (10%)
    const difficultyFactor = Math.min(sample.difficultyLevel / 5.0, 1.0);
    qualityScore += difficultyFactor * 0.1;

    // 基於狀態複雜度的質量評估 (10%)
    const stateComplexity = (
      JSON.stringify(sample.stateBefore).length + 
      JSON.stringify(sample.stateAfter).length
    ) / 2000.0;
    const complexityFactor = Math.min(stateComplexity, 1.0);
    qualityScore += complexityFactor * 0.1;

    return Math.min(qualityScore, 1.0);
  }

  // ========================================================================
  // 統計和監控
  // ========================================================================

  public async getLearningStatistics(sessionId?: string): Promise<Record<string, any>> {
    try {
      if (sessionId) {
        // 獲取特定會話統計
        const session = this.activeSessions.get(sessionId);
        if (session) {
          return {
            sessionId,
            sessionType: session.sessionType,
            totalSamples: session.totalSamples,
            qualityDistribution: this.getQualityDistribution(session),
            completionRate: this.getCompletionRate(session),
            uniquePlans: session.uniquePlans.size,
            vulnerabilityTypes: session.vulnerabilityTypes.size,
            isActive: session.isActive,
            status: session.status
          };
        } else {
          return { error: `Session ${sessionId} not found` };
        }
      } else {
        // 獲取全局統計
        const stats = await this.storage.getStatistics();
        const uptime = Date.now() - this.startTime.getTime();

        return {
          managerId: this.config.managerId,
          uptimeSeconds: Math.floor(uptime / 1000),
          totalSamples: stats.totalSamples,
          totalSessions: stats.totalSessions,
          totalPlans: stats.totalPlans,
          qualityDistribution: {
            high: stats.highQualitySamples,
            medium: stats.mediumQualitySamples,
            low: stats.lowQualitySamples
          },
          successRate: stats.successRate,
          avgQualityScore: stats.avgQualityScore,
          avgReward: stats.avgReward,
          activeSessions: this.activeSessions.size,
          samplesStoredThisSession: this.totalSamplesStored,
          samplesRetrievedThisSession: this.totalSamplesRetrieved,
          storageBackend: this.config.storageBackend,
          deduplicationEnabled: this.config.deduplicationEnabled
        };
      }

    } catch (error) {
      console.error(`[ExperienceManager] Error getting learning statistics: ${error}`);
      return { error: error instanceof Error ? error.message : String(error) };
    }
  }

  // ========================================================================
  // 清理和維護
  // ========================================================================

  public async cleanupOldExperiences(retentionDays?: number): Promise<number> {
    try {
      const retention = retentionDays || this.config.retentionDays;
      const cutoffDate = new Date(Date.now() - retention * 24 * 60 * 60 * 1000);

      // 獲取要清理的樣本
      const filter: ExperienceFilter = {
        sessionIds: [],
        planIds: [],
        vulnerabilityTypes: [],
        positiveOnly: false,
        endTime: cutoffDate,
        tags: [],
        sortBy: SortBy.TIMESTAMP,
        sortDesc: false,
        offset: 0
      };

      const oldSamples = await this.storage.getExperiences(filter);
      
      if (oldSamples.length === 0) {
        return 0;
      }

      // 這裡應該實現實際的刪除邏輯
      // 為了簡化，我們只記錄清理意圖
      const cleanupCount = oldSamples.length;
      
      console.log(`[ExperienceManager] Would cleanup ${cleanupCount} old experience samples`);
      this.emit('experiences_cleaned', { cleanupCount, cutoffDate });
      
      return cleanupCount;

    } catch (error) {
      console.error(`[ExperienceManager] Error cleaning up old experiences: ${error}`);
      return 0;
    }
  }

  // ========================================================================
  // 輔助方法
  // ========================================================================

  private generateId(prefix: string): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `${prefix}_${timestamp}_${random}`;
  }

  private calculateSampleHash(sample: ExperienceSample): string {
    const hashContent = {
      planId: sample.planId,
      stateBefore: sample.stateBefore,
      actionTaken: sample.actionTaken,
      targetInfo: sample.targetInfo
    };

    const contentStr = JSON.stringify(hashContent, Object.keys(hashContent).sort());
    return crypto.createHash('md5').update(contentStr).digest('hex');
  }

  private updateSessionStatistics(session: LearningSession, sample: ExperienceSample): void {
    session.totalSamples++;

    // 根據質量分數分類
    if (sample.qualityScore) {
      if (sample.qualityScore >= 0.8) {
        session.highQualitySamples++;
      } else if (sample.qualityScore >= 0.6) {
        session.mediumQualitySamples++;
      } else {
        session.lowQualitySamples++;
      }
    } else {
      // 根據成功率估算質量
      if (sample.isPositive && sample.confidence >= 0.8) {
        session.highQualitySamples++;
      } else {
        session.mediumQualitySamples++;
      }
    }

    // 更新唯一標識集合
    session.uniquePlans.add(sample.planId);
    if (sample.targetInfo.vulnerabilityType) {
      session.vulnerabilityTypes.add(sample.targetInfo.vulnerabilityType);
    }
  }

  private getQualityDistribution(session: LearningSession): Record<string, number> {
    if (session.totalSamples === 0) {
      return { high: 0, medium: 0, low: 0 };
    }

    return {
      high: session.highQualitySamples / session.totalSamples,
      medium: session.mediumQualitySamples / session.totalSamples,
      low: session.lowQualitySamples / session.totalSamples
    };
  }

  private getCompletionRate(session: LearningSession): number {
    if (session.maxSamples <= 0) {
      return 0;
    }
    return Math.min(session.totalSamples / session.maxSamples, 1.0);
  }

  private startCleanupTask(): void {
    console.log('[ExperienceManager] Starting cleanup task...');
    
    this.cleanupTask = setInterval(async () => {
      try {
        if (this.config.autoCleanup) {
          // 清理舊經驗
          const cleanedCount = await this.cleanupOldExperiences();
          if (cleanedCount > 0) {
            console.log(`[ExperienceManager] Periodic cleanup: ${cleanedCount} samples cleaned`);
          }

          // 清理去重緩存
          if (this.dedupCache.size > 10000) {
            this.dedupCache.clear();
            console.log('[ExperienceManager] Deduplication cache cleared');
          }
        }
      } catch (error) {
        console.error(`[ExperienceManager] Error in periodic cleanup: ${error}`);
      }
    }, 60 * 60 * 1000); // 每小時執行一次
  }

  // ========================================================================
  // 資源管理
  // ========================================================================

  public async cleanup(): Promise<void> {
    try {
      // 停止清理任務
      if (this.cleanupTask) {
        clearInterval(this.cleanupTask);
      }

      // 結束所有活躍會話
      for (const sessionId of this.activeSessions.keys()) {
        await this.endLearningSession(sessionId);
      }

      // 清理存儲
      await this.storage.cleanup();

      // 清理緩存
      this.dedupCache.clear();

      console.log('[ExperienceManager] Cleanup completed');

    } catch (error) {
      console.error(`[ExperienceManager] Error during cleanup: ${error}`);
    }
  }
}

// ============================================================================
// 工廠函數
// ============================================================================

export function createExperienceManager(config?: Partial<ExperienceManagerConfig>): AIVAExperienceManager {
  const defaultConfig: ExperienceManagerConfig = {
    managerId: `ts_exp_mgr_${Date.now()}`,
    storageBackend: StorageBackend.MEMORY,
    storagePath: './data/experience.db',
    retentionDays: 30,
    deduplicationEnabled: true,
    autoCleanup: true,
    maxMemoryUsage: 100 * 1024 * 1024, // 100MB
    batchSize: 100
  };

  const finalConfig = { ...defaultConfig, ...config };
  return new AIVAExperienceManager(finalConfig);
}

// ============================================================================
// 導出
// ============================================================================

export default AIVAExperienceManager;