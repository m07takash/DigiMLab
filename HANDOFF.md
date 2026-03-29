# DigiMLab 引き継ぎ書
> Claude Code への引き継ぎ用ドキュメント

---

## 1. プロジェクト概要

**プロジェクト名:** DigiMLab  
**目的:** 複数のLLMに対してパーソナリティ評価テストを自動実行し、評価レポートを生成するPython環境の構築

---

## 2. 背景と経緯

### やりたいこと
- 複数のAI（社内開発モデル等）に対して、心理学的・言語学的な観点からパーソナリティ評価を自動実施したい
- テスト対象のAIはそれぞれFastAPI等で別環境（別コンテナ）として立てておき、このDigiMLab環境からAPIで呼び出してテストする
- テストセットは順次追加していく方針（論文やGitHubリポジトリを指定して取り込めるようにしたい）
- 評価実行からレポート生成までを自動化する

### 参照した評価手法
以下の手法を調査・整理し、実装優先度を検討した（元資料: `LLMのパーソナリティ評価.xlsx`）

| 優先度 | 手法 | 概要 | 評価される個性 |
|--------|------|------|----------------|
| 1 | AIPsychoBench | 多言語で心理学的な性格診断 | 内面的個性（Big Five等） |
| 2 | MPI | Big Five質問紙（120問）| 統計的個性（OCEAN） |
| 3 | Japanese-RP-Bench | 日本語ロールプレイの一貫性 | 表現的個性（語尾・一人称） |
| 4 | JP-Persona | 日本小説由来の設定再現性 | 社会的個性（役割意識） |
| 5 | CulturalPersonas | 日本文化的バイアス評価 | 文化的個性（空気を読む等） |
| 6 | RPEval | 感情・倫理・一貫性の4軸 | 倫理的個性 |
| 7 | Your Next Token Prediction | 個人の執筆スタイル再現度 | 言語的個性 |
| 8 | CharacterBox | 状況変化下での性格崩れ検証 | 適応的個性 |
| 9 | PersonaGym | 意思決定の思考プロセス評価 | 知的個性 |
| 10 | PersonaLLM | テキストへの性格表出度（人手評価） | 知覚的個性 |

---

## 3. 設計上の決定事項

### アーキテクチャ
- **テスト環境はDockerコンテナ1つ**に集約（`digiml-runner`）
- テスト対象AIは**別コンテナ**として立て、同一Dockerネットワーク（`digiml-net`）経由でHTTP呼び出し
- AIクライアントは `openai_compatible` / `openai` / `anthropic` の3タイプに対応済み

### 設定管理
- APIキー等の秘匿情報は **`system.env`** ファイルで一元管理（環境変数ではなくファイルで管理する方針）
- `env_loader.py` がアプリ起動時に自動ロードし、`${VAR_NAME}` 形式でYAML内から参照できる
- `system.env` は `.gitignore` 対象。`system.env.example` をテンプレートとしてリポジトリ管理する

### 評価方式
- **ルールベース採点**（MPIなど）: 1〜5の回答を正規化してスコア化
- **LLM-as-Judge**（日本語RPなど）: GPT-4o / Claude が4軸（一貫性・自然さ・適切さ・ペルソナ精度）でJSON採点

### テストセット追加
- `python main.py import-github <URL>` でGitHubリポジトリから直接取り込み
- `python main.py import-arxiv <arxiv_id>` でarXiv論文から関連リポジトリを探す

---

## 4. 現在のファイル構成

```
DigiMLab/
├── Dockerfile
├── docker-compose.yml         # ネットワーク: digiml-net, コンテナ: digiml-runner
├── requirements.txt
├── main.py                    # CLIエントリポイント（typer）
├── env_loader.py              # system.env 読み込みモジュール
├── system.env                 # ★ APIキー設定（要編集・Git除外）
├── system.env.example         # テンプレート（Git管理対象）
├── .gitignore
├── SETUP.md                   # 環境構築手順書（Linux向け）
│
├── config/
│   ├── models.yaml            # ★ テスト対象AIのエンドポイント設定
│   └── settings.yaml          # ★ 実行するテストスイートの設定
│
├── evaluators/
│   ├── base.py                # 基底クラス（TestCase / TestResult / EvalReport / BaseEvaluator）
│   ├── mpi_evaluator.py       # MPI（Big Five / OCEAN）実装済み
│   └── rp_evaluator.py        # 日本語ロールプレイ評価（LLM-as-Judge）実装済み
│
├── runners/
│   ├── api_client.py          # 統一AIクライアント
│   └── orchestrator.py        # テスト実行の司令塔（非同期・並列対応）
│
├── importers/
│   ├── github_importer.py     # GitHubからデータセット取り込み
│   └── arxiv_importer.py      # arXivから関連リポジトリ探索
│
├── reporters/
│   └── html_reporter.py       # HTMLレポート生成（JSON出力も兼務）
│
├── datasets/                  # テストデータ置き場（初期は空）
└── reports/                   # レポート出力先（初期は空）
```

---

## 5. 未実装・今後の課題

| 項目 | 優先度 | メモ |
|------|--------|------|
| `datasets/mpi/questions_ja.json` の本番データ整備 | 高 | SETUP.mdに5問のサンプルのみ記載。MPI公式または既存データセットから整備が必要 |
| Web UI（FastAPI）の実装 | 中 | `main.py serve` コマンドは定義済みだが `api.py` が未作成 |
| Excelレポーター | 中 | `config/settings.yaml` に `excel` 形式を定義済みだが未実装 |
| `style_evaluator.py` | 中 | Your Next Token Prediction用。`settings.yaml` に定義済みだが未実装 |
| AIPsychoBench / JP-Persona 等の evaluator | 低〜中 | 手法は調査済み。`BaseEvaluator` を継承して追加する |
| テスト結果のモデル間比較レポート | 低 | 現状は1モデル1レポートの構造 |

---

## 6. 起動・実行コマンド早見表

```bash
# ビルド＆起動
docker compose up --build

# テスト実行（別ターミナルで）
docker compose exec digiml-runner python main.py run
docker compose exec digiml-runner python main.py run --models my_ai

# データセット追加
docker compose exec digiml-runner python main.py import-github <URL> --name <name> --pattern "*.json"
docker compose exec digiml-runner python main.py import-arxiv <arxiv_id>

# 停止
docker compose down
```

---

## 7. 新しい評価手法を追加する手順

1. `evaluators/` に `BaseEvaluator` を継承したクラスを作成
2. `load_dataset()` でJSONを読み込む
3. `score()` でスコアリングロジックを実装
4. `config/settings.yaml` にエントリを追加（`enabled: true` にするだけで自動認識）

```python
# evaluators/my_evaluator.py のひな形
from evaluators.base import BaseEvaluator, TestCase, TestResult

class MyEvaluator(BaseEvaluator):
    def load_dataset(self, path: str) -> list[TestCase]:
        ...
    async def score(self, case: TestCase, result: TestResult) -> TestResult:
        ...
```

```yaml
# config/settings.yaml への追加例
test_suites:
  my_benchmark:
    enabled: true
    description: "新しいベンチマーク"
    dataset: "datasets/my_benchmark/data.json"
    evaluator: "evaluators.my_evaluator.MyEvaluator"
    sample_size: 50
```
