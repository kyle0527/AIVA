"""
HeadlessBrowserPool 使用範例

展示如何使用 HeadlessBrowserPool 管理無頭瀏覽器實例
"""



import asyncio

from .headless_browser_pool import BrowserType, HeadlessBrowserPool, PoolConfig


async def example_basic_usage() -> None:
    """基本使用範例"""
    print("=== 基本使用範例 ===\n")

    # 創建瀏覽器池（默認配置）
    pool = HeadlessBrowserPool()

    try:
        # 初始化池
        await pool.initialize()
        print("[V] 瀏覽器池初始化完成")

        # 使用上下文管理器獲取頁面
        async with pool.get_page() as page:
            if page:  # 真實瀏覽器模式
                await page.goto("https://example.com")
                title = await page.title()
                print(f"[V] 頁面標題: {title}")
            else:  # 模擬模式
                print("[V] 在模擬模式下運行（未安裝 Playwright）")

        # 查看統計
        stats = pool.get_stats()
        print("\n統計信息:")
        print(f"  活動瀏覽器: {stats['active_browsers']}")
        print(f"  活動頁面: {stats['active_pages']}")
        print(f"  總請求數: {stats['total_requests']}")

    finally:
        # 關閉池
        await pool.shutdown()
        print("\n[V] 瀏覽器池已關閉")


async def example_custom_config() -> None:
    """自定義配置範例"""
    print("\n\n=== 自定義配置範例 ===\n")

    # 創建自定義配置
    config = PoolConfig(
        min_instances=1,
        max_instances=5,
        max_pages_per_browser=3,
        browser_type=BrowserType.CHROMIUM,
        headless=True,
        viewport_width=1366,
        viewport_height=768,
        timeout_ms=60000,
    )

    pool = HeadlessBrowserPool(config)

    try:
        await pool.initialize()

        # 獲取配置信息
        stats = pool.get_stats()
        print("配置:")
        print(f"  瀏覽器類型: {stats['config']['browser_type']}")
        print(f"  最大實例數: {stats['config']['max_instances']}")
        print(f"  無頭模式: {stats['config']['headless']}")

    finally:
        await pool.shutdown()


async def example_execute_callback() -> None:
    """使用回調函數範例"""
    print("\n\n=== 回調函數範例 ===\n")

    pool = HeadlessBrowserPool()

    try:
        await pool.initialize()

        # 定義回調函數
        async def scrape_page(page):
            """抓取頁面內容的回調"""
            if page is None:
                return {"mode": "mock", "content": "模擬內容"}

            await page.goto("https://example.com")
            content = await page.content()
            return {
                "mode": "real",
                "url": page.url,
                "content_length": len(content),
            }

        # 執行回調
        result = await pool.execute_on_page("https://example.com", scrape_page)
        print(f"抓取結果: {result}")

    finally:
        await pool.shutdown()


async def example_multiple_pages() -> None:
    """多頁面並發範例"""
    print("\n\n=== 多頁面並發範例 ===\n")

    pool = HeadlessBrowserPool()

    try:
        await pool.initialize()

        # 並發處理多個 URL
        urls = [
            "https://example.com",
            "https://example.org",
            "https://example.net",
        ]

        async def process_url(url: str):
            """處理單個 URL"""
            async with pool.get_page() as page:
                if page:
                    await page.goto(url)
                    return {"url": url, "title": await page.title()}
                else:
                    return {"url": url, "title": f"Mock title for {url}"}

        # 並發執行
        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks)

        print(f"處理了 {len(results)} 個頁面:")
        for result in results:
            print(f"  - {result['url']}: {result['title']}")

        # 查看池狀態
        stats = pool.get_stats()
        print("\n池狀態:")
        print(f"  總頁面創建數: {stats['total_pages_created']}")
        print(f"  總頁面關閉數: {stats['total_pages_closed']}")

    finally:
        await pool.shutdown()


async def example_browser_info() -> None:
    """瀏覽器信息查詢範例"""
    print("\n\n=== 瀏覽器信息查詢範例 ===\n")

    pool = HeadlessBrowserPool()

    try:
        await pool.initialize()

        # 使用頁面（觸發瀏覽器創建）
        async with pool.get_page() as _page:
            pass

        # 列出所有瀏覽器
        browsers = pool.list_browsers()
        print(f"當前有 {len(browsers)} 個瀏覽器實例:")
        for browser in browsers:
            print(f"\n瀏覽器 ID: {browser['browser_id']}")
            print(f"  類型: {browser['browser_type']}")
            print(f"  狀態: {browser['status']}")
            print(f"  頁面數: {browser['page_count']}")
            print(f"  運行時間: {browser['age_seconds']:.1f} 秒")

        # 列出所有頁面
        pages = pool.list_pages()
        print(f"\n當前有 {len(pages)} 個活動頁面")

    finally:
        await pool.shutdown()


async def example_cleanup() -> None:
    """清理空閒瀏覽器範例"""
    print("\n\n=== 清理空閒瀏覽器範例 ===\n")

    config = PoolConfig(
        min_instances=1,
        max_instances=5,
        idle_timeout_seconds=5,  # 5 秒空閒超時
    )

    pool = HeadlessBrowserPool(config)

    try:
        await pool.initialize()

        # 創建多個頁面
        print("創建多個頁面...")
        pages = []
        for _ in range(3):
            async with pool.get_page() as page:
                pages.append(page)
                await asyncio.sleep(0.1)

        print(f"活動瀏覽器: {pool.get_stats()['active_browsers']}")

        # 等待一段時間
        print("\n等待空閒超時...")
        await asyncio.sleep(6)

        # 清理空閒瀏覽器
        closed = await pool.cleanup_idle_browsers()
        print(f"清理了 {closed} 個空閒瀏覽器")
        print(f"剩餘瀏覽器: {pool.get_stats()['active_browsers']}")

    finally:
        await pool.shutdown()


async def example_error_handling() -> None:
    """錯誤處理範例"""
    print("\n\n=== 錯誤處理範例 ===\n")

    pool = HeadlessBrowserPool()

    try:
        # 未初始化就使用會報錯
        try:
            async with pool.get_page() as _page:
                pass
        except RuntimeError as e:
            print(f"[V] 捕獲到預期錯誤: {e}")

        # 正確初始化
        await pool.initialize()

        # 訪問無效的瀏覽器 ID
        try:
            async with pool.get_page(browser_id="invalid_id") as _page:
                pass
        except ValueError as e:
            print(f"[V] 捕獲到預期錯誤: {e}")

    finally:
        await pool.shutdown()


async def main() -> None:
    """運行所有範例"""
    await example_basic_usage()
    await example_custom_config()
    await example_execute_callback()
    await example_multiple_pages()
    await example_browser_info()
    await example_cleanup()
    await example_error_handling()


if __name__ == "__main__":
    asyncio.run(main())
