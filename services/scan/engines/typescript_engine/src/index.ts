/**
 * AIVA Scan Node - Playwright 動態掃描引擎
 * 日期: 2025-10-13
 * 功能: 使用 Playwright 進行動態網頁掃描
 */

import { chromium, Browser } from 'playwright-core';
import * as amqp from 'amqplib';
import { logger } from './utils/logger.js';
import { ScanService } from './services/scan-service.js';
// import { EnhancedDynamicScanService } from './services/enhanced-dynamic-scan.service';
// import { DynamicScanTask, DynamicScanResult } from './interfaces/dynamic-scan.interfaces';

// 遵循 12-factor app 原則獲取 RabbitMQ URL
function getRabbitMQURL(): string {
    // 優先使用完整 URL
    const url = process.env.RABBITMQ_URL;
    if (url) return url;
    
    // 組合式配置
    const host = process.env.RABBITMQ_HOST || 'localhost';
    const port = process.env.RABBITMQ_PORT || '5672';
    const user = process.env.RABBITMQ_USER;
    const password = process.env.RABBITMQ_PASSWORD;
    const vhost = process.env.RABBITMQ_VHOST || '/';
    
    if (!user || !password) {
        throw new Error('RABBITMQ_URL or RABBITMQ_USER/RABBITMQ_PASSWORD must be set');
    }
    
    return `amqp://${user}:${password}@${host}:${port}${vhost}`;
}

const RABBITMQ_URL = getRabbitMQURL();
const TASK_QUEUE = process.env.TASK_QUEUE || 'task.scan.dynamic';
const RESULT_QUEUE = process.env.RESULT_QUEUE || 'findings.new';
// const ENHANCED_TASK_QUEUE = 'task.scan.dynamic.enhanced';

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
