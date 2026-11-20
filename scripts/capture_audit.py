"""Automated browser capture and visual audit for AeroPulse.

Renders and captures desktop (1440x900) and mobile (390x844) views with full WebGL.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def run_audit():
    out_dir = Path("docs/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch Chromium with WebGL enabled
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--enable-webgl",
                "--use-gl=angle",
                "--ignore-gpu-blocklist",
            ],
        )

        # 1. Desktop Audit (1440x900)
        context_desktop = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
        )
        page_d = await context_desktop.new_page()
        print("[Audit] Navigating to http://127.0.0.1:5173/ (Desktop)...")
        await page_d.goto("http://127.0.0.1:5173/", wait_until="networkidle")
        await page_d.wait_for_timeout(3000)  # Wait for Three.js canvas to render

        # Desktop Hero Screenshot
        await page_d.screenshot(path=str(out_dir / "audit_desktop_hero_1440x900.png"))
        print("[Audit] Captured audit_desktop_hero_1440x900.png")

        # Scroll to #machine (Digital Twin)
        await page_d.locator("#machine").scroll_into_view_if_needed()
        await page_d.wait_for_timeout(1500)
        await page_d.screenshot(path=str(out_dir / "audit_desktop_01_machine.png"))
        print("[Audit] Captured audit_desktop_01_machine.png")

        # Scroll to #fleet (Fleet Atlas)
        await page_d.locator("#fleet").scroll_into_view_if_needed()
        await page_d.wait_for_timeout(1000)
        await page_d.screenshot(path=str(out_dir / "audit_desktop_02_fleet.png"))
        print("[Audit] Captured audit_desktop_02_fleet.png")

        # Scroll to #studio (Prediction Studio)
        await page_d.locator("#studio").scroll_into_view_if_needed()
        await page_d.wait_for_timeout(1000)
        await page_d.screenshot(path=str(out_dir / "audit_desktop_03_studio.png"))
        print("[Audit] Captured audit_desktop_03_studio.png")

        # Scroll to #lab (Model Lab)
        await page_d.locator("#lab").scroll_into_view_if_needed()
        await page_d.wait_for_timeout(1000)
        await page_d.screenshot(path=str(out_dir / "audit_desktop_04_lab.png"))
        print("[Audit] Captured audit_desktop_04_lab.png")

        # Scroll to #method (Methodology)
        await page_d.locator("#method").scroll_into_view_if_needed()
        await page_d.wait_for_timeout(1000)
        await page_d.screenshot(path=str(out_dir / "audit_desktop_05_method.png"))
        print("[Audit] Captured audit_desktop_05_method.png")

        # 2. Mobile Audit (390x844 iPhone Viewport)
        context_mobile = await browser.new_context(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            has_touch=True,
            device_scale_factor=2,
        )
        page_m = await context_mobile.new_page()
        print("[Audit] Navigating to http://127.0.0.1:5173/ (Mobile 390x844)...")
        await page_m.goto("http://127.0.0.1:5173/", wait_until="networkidle")
        await page_m.wait_for_timeout(2000)

        # Mobile Hero
        await page_m.screenshot(path=str(out_dir / "audit_mobile_hero_390x844.png"))
        print("[Audit] Captured audit_mobile_hero_390x844.png")

        # Mobile Studio
        await page_m.locator("#studio").scroll_into_view_if_needed()
        await page_m.wait_for_timeout(1000)
        await page_m.screenshot(path=str(out_dir / "audit_mobile_studio_390x844.png"))
        print("[Audit] Captured audit_mobile_studio_390x844.png")

        await browser.close()
        print("[Audit] Visual audit capture complete.")


if __name__ == "__main__":
    asyncio.run(run_audit())
