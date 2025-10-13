/**
 * AIVA Scan Node - Playwright 動態掃描引擎
 * 日期: 2025-10-13
 * 功能: 使用 Playwright 進行動態網頁掃描
 */

import { chromium, Browser } from 'playwright';
import { connect, Connection, Channel } from 'amqplib';
import { logger } from './utils/logger.js';
import { ScanService } from './services/scan-service.js';

const RABBITMQ_URL = process.env.RABBITMQ_URL || 'amqp://aiva:dev_password@localhost:5672/';
const TASK_QUEUE = 'task.scan.dynamic';

interface ScanTask {
  scan_id: string;
  target_url: string;
  max_depth: number;
  max_pages: number;
  enable_javascript: boolean;
}

let browser: Browser | null = null;
let connection: Connection | null = null;
let channel: Channel | null = null;
let scanService: ScanService | null = null;

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

  // 連接 RabbitMQ
  logger.info('📡 連接 RabbitMQ...');
  connection = await connect(RABBITMQ_URL);
  channel = await connection.createChannel();
  await channel.assertQueue(TASK_QUEUE, { durable: true });
  await channel.prefetch(1);
  logger.info('✅ RabbitMQ 已連接');

  logger.info('✅ 初始化完成,開始監聽任務...');
}

async function consumeTasks(): Promise<void> {
  if (!channel || !scanService) {
    throw new Error('Channel 或 ScanService 未初始化');
  }

  await channel.consume(TASK_QUEUE, async (msg) => {
    if (!msg || !channel || !scanService) return;

    try {
      const task: ScanTask = JSON.parse(msg.content.toString());
      logger.info({ scan_id: task.scan_id }, '📥 收到掃描任務');

      // 執行掃描
      const result = await scanService.scan(task);

      logger.info(
        { scan_id: task.scan_id, assets: result.assets.length },
        '✅ 掃描完成'
      );

      // 發送結果到 RabbitMQ
      const resultQueue = 'results.scan.completed';
      await channel.assertQueue(resultQueue, { durable: true });
      await channel.sendToQueue(
        resultQueue,
        Buffer.from(JSON.stringify(result)),
        { persistent: true }
      );

      // 確認訊息
      channel.ack(msg);
    } catch (error) {
      logger.error({ error }, '❌ 處理任務失敗');
      // 拒絕訊息並重新排隊
      if (msg && channel) {
        channel.nack(msg, false, true);
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

  if (channel) {
    await channel.close();
    logger.info('✅ Channel 已關閉');
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
