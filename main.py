"""
DigiMLab — CLI

使い方:
  python main.py run                            # 全モデル×全テスト実行
  python main.py run --models model_a gpt4o    # 特定モデルのみ
  python main.py import-github <URL>           # GitHubからデータセット追加
  python main.py import-arxiv <ID>             # arXiv論文からリポジトリを探す
  python main.py serve                          # Web UIを起動
"""
import typer
from typing import Optional

app = typer.Typer(help="DigiMLab")


@app.command()
def run(
    models: Optional[list[str]] = typer.Option(None, "--models", "-m", help="実行するモデルID（省略時は全て）"),
    config: str = typer.Option("config/models.yaml", "--config"),
    settings: str = typer.Option("config/settings.yaml", "--settings"),
):
    """テストを実行し、評価レポートを生成する"""
    from runners.orchestrator import Orchestrator
    orch = Orchestrator(models_config_path=config, settings_path=settings)
    orch.run(model_ids=models or None)


@app.command("import-github")
def import_github(
    repo_url: str = typer.Argument(..., help="GitHubリポジトリURL"),
    name: str = typer.Option(..., "--name", "-n", help="データセット名"),
    pattern: str = typer.Option("*.json", "--pattern", "-p", help="ファイルパターン"),
):
    """GitHubリポジトリからテストセットをインポートする"""
    from importers.github_importer import GitHubImporter
    importer = GitHubImporter()
    out = importer.import_repo(repo_url=repo_url, dataset_name=name, file_pattern=pattern)
    typer.echo(f"✅ インポート完了: {out}")
    typer.echo("次に config/settings.yaml へエントリを追加してください。")
    typer.echo(f"  evaluator: evaluators.base.BaseEvaluator  # カスタム評価器クラスに変更してください")


@app.command("import-arxiv")
def import_arxiv(
    arxiv_id: str = typer.Argument(..., help="arXiv ID (例: 2509.16530)"),
):
    """arXiv論文からデータセットのインポート手順を表示する"""
    from importers.arxiv_importer import ArxivImporter
    importer = ArxivImporter()
    steps = importer.suggest_import_steps(arxiv_id)
    typer.echo(steps)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port"),
):
    """Web UI（FastAPI）を起動する"""
    import uvicorn
    from api import create_app
    app_instance = create_app()
    uvicorn.run(app_instance, host=host, port=port)


if __name__ == "__main__":
    app()
