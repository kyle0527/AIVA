"""
DynamicContentExtractor 使用範例

展示如何使用 DynamicContentExtractor 提取動態內容
"""

from __future__ import annotations

import asyncio

from .dynamic_content_extractor import (
    ContentType,
    DynamicContentExtractor,
    ExtractionConfig,
)
from .headless_browser_pool import HeadlessBrowserPool


async def example_basic_usage() -> None:
    """基本使用範例"""
    print("=== 基本使用範例 ===\n")

    # 創建提取器
    extractor = DynamicContentExtractor()
    print("[OK] 創建提取器")

    # 模擬模式（無頁面對象）
    contents = await extractor.extract_from_url("https://example.com")
    print(f"[OK] 提取了 {len(contents)} 個內容（模擬模式）")

    # 查看統計
    stats = extractor.get_stats()
    print("\n統計信息:")
    print(f"  總內容數: {stats['total_contents']}")
    print(f"  網絡請求數: {stats['total_network_requests']}")


async def example_with_browser_pool() -> None:
    """與瀏覽器池集成的範例"""
    print("\n\n=== 與瀏覽器池集成範例 ===\n")

    # 創建瀏覽器池
    pool = HeadlessBrowserPool()
    await pool.initialize()

    # 創建提取器
    extractor = DynamicContentExtractor()

    try:
        # 使用瀏覽器池
        async with pool.get_page() as page:
            # 從頁面提取內容
            contents = await extractor.extract_from_url(
                "https://example.com", page=page
            )

            print(f"提取了 {len(contents)} 個動態內容")

            # 按類型統計
            stats = extractor.get_stats()
            print("\n按類型統計:")
            for content_type, count in stats["contents_by_type"].items():
                print(f"  {content_type}: {count}")

    finally:
        await pool.shutdown()


async def example_custom_config() -> None:
    """自定義配置範例"""
    print("\n\n=== 自定義配置範例 ===\n")

    # 創建自定義配置
    config = ExtractionConfig(
        extract_forms=True,
        extract_links=True,
        extract_ajax=True,
        extract_api_calls=True,
        extract_js_variables=False,  # 禁用 JS 變量提取
        extract_event_listeners=False,  # 禁用事件監聽器提取
        wait_for_network_idle=True,
        network_idle_timeout_ms=2000,
        capture_network_requests=True,
    )

    _extractor = DynamicContentExtractor(config)

    print("配置:")
    print(f"  提取表單: {config.extract_forms}")
    print(f"  提取鏈接: {config.extract_links}")
    print(f"  提取 AJAX: {config.extract_ajax}")
    print(f"  網絡空閒超時: {config.network_idle_timeout_ms}ms")


async def example_extract_by_type() -> None:
    """按類型提取範例"""
    print("\n\n=== 按類型提取範例 ===\n")

    pool = HeadlessBrowserPool()
    await pool.initialize()

    extractor = DynamicContentExtractor()

    try:
        async with pool.get_page() as page:
            # 提取內容
            await extractor.extract_from_url("https://example.com", page=page)

            # 按類型獲取
            forms = extractor.get_contents_by_type(ContentType.FORM)
            links = extractor.get_contents_by_type(ContentType.LINK)
            ajax_endpoints = extractor.get_contents_by_type(ContentType.AJAX_ENDPOINT)

            print(f"表單: {len(forms)}")
            print(f"鏈接: {len(links)}")
            print(f"AJAX 端點: {len(ajax_endpoints)}")

            # 查看第一個表單
            if forms:
                form = forms[0]
                print("\n第一個表單:")
                print(f"  URL: {form.url}")
                print(f"  方法: {form.attributes.get('method')}")
                print(f"  輸入欄位數: {len(form.attributes.get('inputs', []))}")

    finally:
        await pool.shutdown()


async def example_network_requests() -> None:
    """網絡請求監控範例"""
    print("\n\n=== 網絡請求監控範例 ===\n")

    pool = HeadlessBrowserPool()
    await pool.initialize()

    # 啟用網絡請求捕獲
    config = ExtractionConfig(capture_network_requests=True)
    extractor = DynamicContentExtractor(config)

    try:
        async with pool.get_page() as page:
            await extractor.extract_from_url("https://example.com", page=page)

            # 獲取網絡請求
            requests = extractor.get_network_requests()
            print(f"捕獲了 {len(requests)} 個網絡請求")

            # 顯示前幾個請求
            for i, req in enumerate(requests[:5], 1):
                print(f"\n請求 {i}:")
                print(f"  URL: {req.url}")
                print(f"  方法: {req.method}")
                print(f"  類型: {req.resource_type}")
                if req.response_status:
                    print(f"  狀態: {req.response_status}")

    finally:
        await pool.shutdown()


async def example_convert_to_assets() -> None:
    """轉換為 Asset 對象範例"""
    print("\n\n=== 轉換為 Asset 範例 ===\n")

    pool = HeadlessBrowserPool()
    await pool.initialize()

    extractor = DynamicContentExtractor()

    try:
        async with pool.get_page() as page:
            # 提取內容
            contents = await extractor.extract_from_url(
                "https://example.com", page=page
            )

            # 轉換為 Asset 對象
            assets = extractor.convert_to_assets(contents)
            print(f"轉換了 {len(assets)} 個 Asset")

            # 顯示 Asset 信息
            for asset in assets[:3]:
                print("\nAsset:")
                print(f"  ID: {asset.asset_id}")
                print(f"  類型: {asset.type}")
                print(f"  值: {asset.value}")
                if asset.parameters:
                    print(f"  參數: {asset.parameters}")

    finally:
        await pool.shutdown()


async def example_after_interaction() -> None:
    """互動後提取範例"""
    print("\n\n=== 互動後提取範例 ===\n")

    pool = HeadlessBrowserPool()
    await pool.initialize()

    extractor = DynamicContentExtractor()

    try:
        async with pool.get_page() as page:
            if page:
                # 訪問頁面
                await page.goto("https://example.com")

                # 執行某些互動（例如點擊）
                # await page.click("#some-button")

                # 在互動後提取內容
                contents = await extractor.extract_after_interaction(
                    page, "https://example.com", wait_time_ms=1000
                )

                print(f"互動後提取了 {len(contents)} 個內容")

    finally:
        await pool.shutdown()


async def example_clear_and_reuse() -> None:
    """清空和重用範例"""
    print("\n\n=== 清空和重用範例 ===\n")

    extractor = DynamicContentExtractor()

    # 第一次使用
    await extractor.extract_from_url("https://example.com")
    print(f"第一次提取: {len(extractor.get_extracted_contents())} 個內容")

    # 清空
    extractor.clear()
    print(f"清空後: {len(extractor.get_extracted_contents())} 個內容")

    # 重用
    await extractor.extract_from_url("https://example.org")
    print(f"重用後: {len(extractor.get_extracted_contents())} 個內容")


async def main() -> None:
    """運行所有範例"""
    await example_basic_usage()
    await example_custom_config()
    await example_clear_and_reuse()

    # 需要 Playwright 的範例
    # await example_with_browser_pool()
    # await example_extract_by_type()
    # await example_network_requests()
    # await example_convert_to_assets()
    # await example_after_interaction()


if __name__ == "__main__":
    asyncio.run(main())
