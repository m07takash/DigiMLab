"""DigiMLab — Web UI (FastAPI)"""
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
import json

from runners.orchestrator import Orchestrator


def create_app() -> FastAPI:
    app = FastAPI(title="DigiMLab", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        reports_dir = Path("reports")
        reports = sorted(reports_dir.glob("report_*.html"), reverse=True)
        items = "".join(
            f'<li><a href="/reports/{r.name}">{r.stem}</a></li>' for r in reports
        )
        return f"""<!DOCTYPE html>
<html lang="ja">
<head><meta charset="UTF-8"><title>DigiMLab</title>
<style>body{{font-family:sans-serif;margin:40px;}}h1{{color:#5a4abe;}}</style></head>
<body>
<h1>DigiMLab</h1>
<h2>Actions</h2>
<form method="post" action="/run"><button type="submit">Run All Tests</button></form>
<h2>Reports</h2>
<ul>{items if items else "<li>No reports yet</li>"}</ul>
</body></html>"""

    @app.get("/reports/{filename}")
    async def get_report(filename: str):
        path = Path("reports") / filename
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        if filename.endswith(".html"):
            return FileResponse(path, media_type="text/html")
        return FileResponse(path)

    @app.post("/run")
    async def run_tests(background_tasks: BackgroundTasks):
        def _run():
            orch = Orchestrator()
            orch.run()
        background_tasks.add_task(_run)
        return HTMLResponse(
            '<html><body><p>Tests started. <a href="/">Back to top</a></p></body></html>'
        )

    @app.get("/api/reports")
    async def list_reports():
        reports_dir = Path("reports")
        files = sorted(reports_dir.glob("report_*.json"), reverse=True)
        results = []
        for f in files[:20]:
            results.append({"name": f.stem, "file": f.name})
        return results

    @app.get("/api/reports/{filename}")
    async def get_report_json(filename: str):
        path = Path("reports") / filename
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return json.loads(path.read_text(encoding="utf-8"))

    return app
