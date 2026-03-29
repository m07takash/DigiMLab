"""HTMLレポートジェネレーター"""
from pathlib import Path
from datetime import datetime
from evaluators.base import EvalReport

TEMPLATE = """\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>DigiMLab 評価レポート</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 24px; background: #f5f5f5; color: #333; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 32px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card h2 {{ font-size: 17px; margin: 0 0 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f0f0f0; padding: 8px 12px; text-align: left; font-weight: 500; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  .score-bar {{ height: 8px; border-radius: 4px; background: #e0e0e0; margin-top: 4px; }}
  .score-fill {{ height: 100%; border-radius: 4px; background: #5a4abe; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }}
  .summary-item {{ background: #f8f8f8; border-radius: 8px; padding: 12px; text-align: center; }}
  .summary-item .val {{ font-size: 24px; font-weight: 600; color: #5a4abe; }}
  .summary-item .lbl {{ font-size: 11px; color: #888; margin-top: 4px; }}
  .error {{ color: #c0392b; font-size: 12px; }}
</style>
</head>
<body>
<h1>DigiMLab 評価レポート</h1>
<div class="meta">生成日時: {generated_at}</div>

{suite_sections}

</body>
</html>
"""

SUITE_SECTION = """\
<div class="card">
  <h2>{suite_name} — {model_name}</h2>
  <div class="summary-grid">
    {summary_items}
  </div>
  <br>
  <table>
    <thead><tr><th>ID</th><th>スコア</th><th>詳細</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>
"""


def _summary_items(summary: dict) -> str:
    items = []
    for k, v in summary.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, dict) and "mean" in sv:
                    val = f"{sv['mean']:.2f}" if sv["mean"] is not None else "-"
                    items.append(f'<div class="summary-item"><div class="val">{val}</div><div class="lbl">{sv.get("label", sk)}</div></div>')
                elif isinstance(sv, (int, float)):
                    items.append(f'<div class="summary-item"><div class="val">{sv:.3f}</div><div class="lbl">{sk}</div></div>')
        elif isinstance(v, (int, float)):
            items.append(f'<div class="summary-item"><div class="val">{v}</div><div class="lbl">{k}</div></div>')
    return "\n".join(items)


def _result_row(r) -> str:
    if r.error:
        return f'<tr><td>{r.case_id}</td><td class="error">エラー</td><td class="error">{r.error}</td></tr>'
    score_pct = int((r.score or 0) * 100)
    detail_str = ", ".join(f"{k}={v}" for k, v in r.score_detail.items()) if r.score_detail else ""
    bar = f'<div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>'
    return f'<tr><td>{r.case_id}</td><td>{r.score:.3f if r.score is not None else "-"}{bar}</td><td>{detail_str}</td></tr>'


class HTMLReporter:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, reports: list[EvalReport], run_id: str) -> Path:
        sections = []
        for rep in reports:
            rows = "\n".join(_result_row(r) for r in rep.results)
            sections.append(SUITE_SECTION.format(
                suite_name=rep.suite_name,
                model_name=rep.model_name,
                summary_items=_summary_items(rep.summary),
                rows=rows,
            ))

        html = TEMPLATE.format(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            suite_sections="\n".join(sections),
        )
        out = self.output_dir / f"report_{run_id}.html"
        out.write_text(html, encoding="utf-8")
        print(f"[Report] HTML → {out}")
        return out
