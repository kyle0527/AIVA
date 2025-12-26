// Worker Pool - 高效並發任務執行
package common

import (
	"context"
	"log"
	"sync"
	"time"
)

// WorkerPool - 並發執行池
type WorkerPool struct {
	workers    int
	taskQueue  chan *Task
	resultChan chan *TaskResult
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
}

// NewWorkerPool - 創建 Worker Pool
func NewWorkerPool(workers int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())

	pool := &WorkerPool{
		workers:    workers,
		taskQueue:  make(chan *Task, workers*2),       // 緩衝隊列，減少阻塞
		resultChan: make(chan *TaskResult, workers*2), // 緩衝結果通道
		ctx:        ctx,
		cancel:     cancel,
	}

	// 啟動 Workers
	for i := 0; i < workers; i++ {
		pool.wg.Add(1)
		go pool.worker(i)
	}

	log.Printf("[INFO] Worker pool started: workers=%d\n", workers)

	return pool
}

// worker - Worker 協程
func (p *WorkerPool) worker(id int) {
	defer p.wg.Done()

	log.Printf("[DEBUG] Worker started: id=%d\n", id)

	for {
		select {
		case <-p.ctx.Done():
			log.Printf("[DEBUG] Worker stopped: id=%d\n", id)
			return

		case task, ok := <-p.taskQueue:
			if !ok {
				log.Printf("[DEBUG] Worker exiting: id=%d (channel closed)\n", id)
				return
			}

			// 執行任務
			startTime := time.Now()
			assets, err := task.Scanner.Scan(p.ctx, []string{task.Target}, task.Config)
			duration := time.Since(startTime)

			// 發送結果
			result := &TaskResult{
				TaskID:   task.ID,
				Target:   task.Target,
				Assets:   assets,
				Error:    err,
				Duration: duration,
			}

			select {
			case p.resultChan <- result:
				// 結果已發送
			case <-p.ctx.Done():
				return
			}
		}
	}
}

// Submit - 提交任務
func (p *WorkerPool) Submit(task *Task) error {
	select {
	case p.taskQueue <- task:
		return nil
	case <-p.ctx.Done():
		return context.Canceled
	}
}

// Results - 獲取結果通道
func (p *WorkerPool) Results() <-chan *TaskResult {
	return p.resultChan
}

// Close - 關閉 Worker Pool（不等待完成）
func (p *WorkerPool) Close() {
	close(p.taskQueue)
	p.wg.Wait()
	close(p.resultChan)
}

// Shutdown - 優雅關閉（等待完成或超時）
func (p *WorkerPool) Shutdown(timeout time.Duration) error {
	log.Printf("[INFO] Worker pool shutting down: timeout=%v\n", timeout)

	// 關閉任務隊列（不再接受新任務）
	close(p.taskQueue)

	// 等待所有任務完成或超時
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("[INFO] Worker pool shutdown complete")
		p.cancel()
		close(p.resultChan)
		return nil
	case <-time.After(timeout):
		log.Println("[WARN] Worker pool shutdown timeout, forcing stop")
		p.cancel()
		close(p.resultChan)
		return context.DeadlineExceeded
	}
}

