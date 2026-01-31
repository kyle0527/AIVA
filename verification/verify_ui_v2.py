from playwright.sync_api import sync_playwright
import os
import time

def run():
    file_path = os.path.abspath("web/index_v3.html")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{file_path}")

        print("Page loaded.")

        # Click the first module button
        modules = page.locator(".bizlogic-module")
        first_module_btn = modules.first.locator("button")
        first_module_btn.click()
        print("Clicked module button.")

        # Wait for modal
        modal = page.locator("#runModuleModal")
        modal.wait_for(state="visible")
        print("Modal visible.")

        # Wait a bit for transition
        time.sleep(1)

        # Take screenshot
        page.screenshot(path="verification/modal_screenshot.png")
        print("Screenshot saved to verification/modal_screenshot.png")

        browser.close()

if __name__ == "__main__":
    run()
