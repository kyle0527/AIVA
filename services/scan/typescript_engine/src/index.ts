/**
 * AIVA Scan Node - Playwright 動態掃描引擎
 * 日期: 2025-10-13
 * 功能: 使用 Playwright 進行動態網頁掃描
 */

import { chromium, Browser } from 'playwright-core';
import * as amqp from 'amqplib';
import { logger } from './utils/logger.js';
import { ScanService } from './services/scan-service.js';
import { DOMSecurityAnalyzer } from './dom-security-analyzer.js';
// import { EnhancedDynamicScanService } from './services/enhanced-dynamic-scan.service';
// import { DynamicScanTask, DynamicScanResult } from './interfaces/dynamic-scan.interfaces';

// 環境變數配置 - 開箱即用，生產環境時才需要覆蓋
const RABBITMQ_URL = process.env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672/';
const TASK_QUEUE = 'task.scan.dynamic';  // 固定使用標準隊列名稱
const RESULT_QUEUE = 'findings.new';      // 固定使用標準隊列名稱

interface ScanTask {
  scan_id: string;
  target_url: string;
  max_depth: number;
  max_pages: number;
  enable_javascript: boolean;
}

let browser: Browser | null = null;
let connection: amqp.Channel | null = null;
let scanService: ScanService | null = null;
// let enhancedScanService: EnhancedDynamicScanService | null = null;

async function initialize(): Promise<void> {
  logger.info('🚀 初始化 AIVA Scan Node...');

  // 啟動瀏覽器
  logger.info('🌐 啟動 Chromium 瀏覽器...');
  browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  logger.info('✅ 瀏覽器已啟動');

  // 初始化掃描服務
  scanService = new ScanService(browser);
  // enhancedScanService = new EnhancedDynamicScanService(browser);

  // 連接 RabbitMQ (使用官方 Promise API)
  logger.info('📡 連接 RabbitMQ...');
  const conn = await amqp.connect(RABBITMQ_URL);
  const channel = await conn.createChannel();
  connection = channel;
  await connection.assertQueue(TASK_QUEUE, { durable: true });
  await connection.prefetch(1);
  logger.info('✅ RabbitMQ 已連接');

  logger.info('✅ 初始化完成,開始監聽任務...');
}

async function consumeTasks(): Promise<void> {
  if (!connection || !scanService) {
    throw new Error('Connection 或 ScanService 未初始化');
  }

  await connection.consume(TASK_QUEUE, async (msg) => {
    if (!msg || !connection || !scanService) return;

    try {
      const task: ScanTask = JSON.parse(msg.content.toString());
      logger.info({ scan_id: task.scan_id }, '📥 收到掃描任務');

      // 執行掃描
      const result = await scanService.scan(task);

      // 執行 DOM 安全分析（如果啟用 JavaScript）
      if (task.enable_javascript && browser) {
        logger.info({ scan_id: task.scan_id }, '🔍 執行 DOM 安全分析...');
        const page = await browser.newPage();
        const domAnalyzer = new DOMSecurityAnalyzer(page);
        
        try {
          const domFindings = await domAnalyzer.analyze(task.target_url);
          
          // 將 DOM 發現添加到結果中
          if (domFindings.length > 0) {
            logger.info(
              { scan_id: task.scan_id, findings: domFindings.length },
              '✅ DOM 安全分析完成'
            );
            
            // 將發現添加到結果（可以擴展 result 結構）
            (result as any).dom_security_findings = domFindings;
          }
        } catch (error) {
          logger.error({ error }, '❌ DOM 安全分析失敗');
        } finally {
          await page.close();
        }
      }

      logger.info(
        { scan_id: task.scan_id, assets: result.assets.length },
        '✅ 掃描完成'
      );

      // 發送結果到 RabbitMQ (統一隊列命名標準)
      await connection.assertQueue(RESULT_QUEUE, { durable: true });
      await connection.sendToQueue(
        RESULT_QUEUE,
        Buffer.from(JSON.stringify(result)),
        { persistent: true }
      );

      // 確認訊息
      connection.ack(msg);
    } catch (error) {
      logger.error({ error }, '❌ 處理任務失敗');
      // 拒絕訊息並重新排隊
      if (msg && connection) {
        connection.nack(msg, false, true);
      }
    }
  });
}

async function shutdown(): Promise<void> {
  logger.info('🛑 關閉服務...');

  if (browser) {
    await browser.close();
    logger.info('✅ 瀏覽器已關閉');
  }

  if (connection) {
    await connection.close();
    logger.info('✅ RabbitMQ 連接已關閉');
  }

  process.exit(0);
}

// 優雅關閉
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// 主程序
async function main(): Promise<void> {
  try {
    await initialize();
    await consumeTasks();
  } catch (error) {
    logger.error({ error }, '❌ 啟動失敗');
    process.exit(1);
  }
}

main();
