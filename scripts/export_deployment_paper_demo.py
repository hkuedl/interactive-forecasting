"""Export the fixture-backed Deployment UI as PNG and PDF with installed Chrome."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BROWSER = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")
DEFAULT_OUTPUT = ROOT / "docs/figures/paper_demo"
FILES = {
    "/": ROOT / "web/index.html",
    "/static/styles.css": ROOT / "web/styles.css",
    "/static/deployment_paper_demo.css": ROOT / "web/deployment_paper_demo.css",
    "/static/app.js": ROOT / "web/app.js",
    "/static/training.js": ROOT / "web/training.js",
    "/static/deployment.js": ROOT / "web/deployment.js",
    "/static/deployment_paper_demo.js": ROOT / "web/deployment_paper_demo.js",
    "/data/examples/deployment_paper_demo.json": ROOT / "data/examples/deployment_paper_demo.json",
}


class DemoHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        return str(FILES.get(urlsplit(path).path, ROOT / "missing-demo-resource"))

    def log_message(self, format: str, *args: object) -> None:
        pass


def winpath(path: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def windows_temp() -> Path:
    result = subprocess.run(["cmd.exe", "/c", "echo", "%TEMP%"], capture_output=True, check=True)
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    windows = next((line.strip() for line in reversed(lines) if ":\\" in line), None)
    if not windows:
        raise RuntimeError("Could not locate Windows temporary directory")
    return Path(subprocess.check_output(["wslpath", "-u", windows], text=True).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, default=DEFAULT_BROWSER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=2100)
    parser.add_argument("--height", type=int, default=3000)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()
    if not args.browser.is_file():
        raise SystemExit("Browser not available: " + str(args.browser))
    if not FILES["/data/examples/deployment_paper_demo.json"].is_file():
        raise SystemExit("Build the fixture first with scripts/build_deployment_paper_fixture.py")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/?paper-demo=1"
    try:
        with tempfile.TemporaryDirectory(prefix="iforecast-figure-", dir=windows_temp()) as temp:
            folder = Path(temp)
            common = [
                str(args.browser),
                "--headless=new",
                "--no-first-run",
                "--disable-gpu",
                "--hide-scrollbars",
                "--virtual-time-budget=10000",
                "--user-data-dir=" + winpath(folder / "profile"),
                f"--window-size={args.width},{args.height}",
                f"--force-device-scale-factor={args.scale}",
            ]
            dom = subprocess.run(
                [*common, "--dump-dom", url], capture_output=True, timeout=60, check=True
            )
            rendered = dom.stdout.decode("utf-8", errors="replace")
            for required in (
                'data-paper-ready="true"',
                "Calendar references",
                "Weather analogs",
                "August 21, 2014",
                "GEFCom2014",
                "Task Manager",
                "D-365",
                "Adjusted forecast",
                "Original forecast",
                "Similarity score",
                "Temperature profiles",
                "Historical load profiles",
                "Can we preview a 2% increase",
                'id="chat-input"',
                'data-paper-bottom="',
            ):
                if required not in rendered or "Paper demo failed:" in rendered:
                    raise RuntimeError(
                        "Demo render check failed: "
                        + required
                        + "\n"
                        + dom.stderr.decode("utf-8", errors="replace")[-1000:]
                    )
            target = folder / "deployment_paper_demo.png"
            subprocess.run(
                [*common, "--screenshot=" + winpath(target), url],
                capture_output=True,
                timeout=60,
                check=True,
            )
            if not target.is_file():
                raise RuntimeError("Browser did not write the screenshot")
            match = re.search(r'data-paper-bottom="([0-9.]+)"', rendered)
            if match is None:
                raise RuntimeError("Rendered weather-section boundary unavailable")
            with Image.open(target) as raw_image:
                bottom = round((float(match.group(1)) + 26) * raw_image.height / args.height)
                if not 0 < bottom <= raw_image.height:
                    raise RuntimeError("Rendered page exceeds screenshot height")
                cropped = raw_image.crop((0, 0, raw_image.width, bottom))
                cropped.save(output / "deployment_paper_demo.png")
                cropped.save(output / "deployment_paper_demo.pdf", "PDF", resolution=200.0)
            with Image.open(output / "deployment_paper_demo.png") as screenshot_image:
                factor_x = screenshot_image.width / args.width
                factor_y = screenshot_image.height / args.height
                for section, suffix in (
                    ("references", "references"),
                    ("weather", "weather_analogs"),
                ):
                    match = re.search(r"data-paper-crop-" + section + r'="([0-9.,]+)"', rendered)
                    if match is None:
                        raise RuntimeError("Rendered card bounds unavailable: " + section)
                    left, top, width, height = map(float, match.group(1).split(","))
                    box = (
                        max(0, round((left - 8) * factor_x)),
                        max(0, round((top - 8) * factor_y)),
                        min(screenshot_image.width, round((left + width + 8) * factor_x)),
                        min(screenshot_image.height, round((top + height + 8) * factor_y)),
                    )
                    screenshot_image.crop(box).save(output / f"deployment_paper_demo_{suffix}.png")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    print("PNG:", output / "deployment_paper_demo.png")
    print("PDF:", output / "deployment_paper_demo.pdf")


if __name__ == "__main__":
    main()
