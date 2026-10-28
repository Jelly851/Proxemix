"""Browser-level proof for the stage-10 demo UI — separate from
smoke_test.py (which only hits the API, no browser) per the build brief.
Launches the real server as a subprocess, drives it with Playwright
(headless Chromium), and checks the one handler every citation/event
click is supposed to share: seeking the video.

Run: py ui_test.py
Screenshots land in outputs/ui/.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(HERE, "data", "demo_bundle")
SCREENSHOT_DIR = os.path.join(HERE, "outputs", "ui")
PORT = 8731
BASE_URL = f"http://127.0.0.1:{PORT}"
FPS = 25.0


def wait_for_server(url, timeout_s=30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "proxemix.serve", "--bundle", BUNDLE_DIR, "--port", str(PORT)],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not wait_for_server(f"{BASE_URL}/api/bundle"):
            out = server.stdout.read() if server.stdout else ""
            raise RuntimeError(f"server never came up:\n{out}")

        events = json.loads(urllib.request.urlopen(f"{BASE_URL}/api/events").read())
        target_event = next(
            e for e in events
            if e["type"] == "zone_first_entry" and e["actor_ids"] == "P4_rightzone_entry"
        )
        expected_time_event = (target_event["frame_start"] + 0.5) / FPS

        chat_resp = json.loads(urllib.request.urlopen(
            urllib.request.Request(
                f"{BASE_URL}/api/chat", method="POST",
                data=json.dumps({"question": "what is the slowest zone entry?"}).encode(),
                headers={"Content-Type": "application/json"},
            )
        ).read())
        target_citation = chat_resp["citations"][0]
        expected_time_citation = (target_citation["frame_start"] + 0.5) / FPS

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 700})
            page.goto(BASE_URL, wait_until="networkidle")

            page.wait_for_selector("#event-list li")
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "01_initial_player.png"))
            print("screenshot: 01_initial_player.png")

            event_li = page.locator("#event-list li", has_text=target_event["label"])
            event_li.click()
            page.wait_for_timeout(200)
            got_time = page.evaluate("document.getElementById('video').currentTime")
            print(f"event click -> video.currentTime={got_time:.3f} (expected {expected_time_event:.3f})")
            assert abs(got_time - expected_time_event) < 0.05, \
                f"event click did not seek correctly: got {got_time}, expected {expected_time_event}"
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "02_event_seek_overlay.png"))
            print("screenshot: 02_event_seek_overlay.png")

            page.click("button[data-tab='chat']")
            page.fill("#chat-input", "what is the slowest zone entry?")
            page.click("#chat-form button[type=submit]")
            page.wait_for_selector(".chat-msg.a .cite-btn")
            page.click(".chat-msg.a .cite-btn")
            page.wait_for_timeout(200)
            got_time_chat = page.evaluate("document.getElementById('video').currentTime")
            print(f"chat citation click -> video.currentTime={got_time_chat:.3f} "
                  f"(expected {expected_time_citation:.3f})")
            assert abs(got_time_chat - expected_time_citation) < 0.05, \
                f"chat citation click did not seek correctly: got {got_time_chat}, expected {expected_time_citation}"
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "03_chat_citation_seek.png"))
            print("screenshot: 03_chat_citation_seek.png")

            browser.close()

        print("\nUI_TEST PASSED")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
