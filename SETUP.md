# DigiMLab 環境 — 構築手順書

---

## 前提：必要なもの

| 項目 | 内容 |
|------|------|
| Docker Desktop | コンテナ実行環境（無料） |
| Git | ファイル管理（無料） |
| APIキー | OpenAI または Anthropic（どちらか一方でOK） |
| ターミナル | Mac: ターミナル.app / Win: PowerShell |

---

## STEP 1 — Docker Desktop をインストール

### Mac（Apple Silicon / Intel 共通）

1. https://www.docker.com/products/docker-desktop/ を開く
2. **「Download for Mac」** をクリック
   - Apple Silicon（M1〜）→ **Apple Chip** 版を選択
   - Intel Mac → **Intel Chip** 版を選択
3. ダウンロードした `.dmg` を開き、Docker.app を **Applications** フォルダへドラッグ
4. Docker.app を起動（メニューバーにクジラアイコンが出れば成功）
5. ターミナルで確認：

```bash
docker --version
docker compose version
```

### Windows 11

1. **WSL2 を有効化**（管理者権限の PowerShell で実行）：

```powershell
wsl --install
```

インストール後、**PCを再起動**する。

2. 再起動後、Ubuntu が自動で起動するのでユーザー名・パスワードを設定
3. https://www.docker.com/products/docker-desktop/ を開く
4. **「Download for Windows」** をクリックしてインストーラを実行
5. インストール中に **「Use WSL 2 instead of Hyper-V」** にチェックが入っていることを確認
6. インストール完了後、Docker Desktop を起動
7. PowerShell で確認：

```powershell
docker --version
docker compose version
```

### Linux（Ubuntu 22.04 / 24.04）

```bash
# 古いバージョンを削除
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null

# 公式スクリプトでインストール
curl -fsSL https://get.docker.com | sh

# 一般ユーザーで docker を使えるようにする
sudo usermod -aG docker $USER

# 一度ログアウト→ログインし直してから確認
docker --version
docker compose version
```

---

## STEP 2 — Git をインストール

### Mac

```bash
# Homebrewがない場合はまず下記を実行
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install git
git --version
```

### Windows 11

1. https://git-scm.com/download/win を開いてインストーラを実行
2. すべての設定はデフォルトのままでOK
3. PowerShell を**新しく開いてから**確認：

```powershell
git --version
```

### Linux（Ubuntu）

```bash
sudo apt update && sudo apt install -y git
git --version
```

---

## STEP 3 — プロジェクトを配置する

ダウンロードしたプロジェクトフォルダ（`DigiMLab`）を
作業したいディレクトリに置いて、ターミナルで移動します。

### Mac / Linux

```bash
# ホームディレクトリ直下に置いた場合
cd ~/DigiMLab
```

### Windows（PowerShell）

```powershell
# デスクトップに置いた場合
cd $HOME\Desktop\DigiMLab
```

配置が正しければ、`ls`（Mac/Linux）または `dir`（Windows）で
以下のファイルが見えます：

```
Dockerfile
docker-compose.yml
main.py
system.env.example
system.env        ← ★次のSTEPで作成します
config/
datasets/
evaluators/
...
```

---

## STEP 4 — APIキーを設定する（system.env）

### ファイルを作成

**Mac / Linux：**

```bash
cp system.env.example system.env
```

**Windows（PowerShell）：**

```powershell
Copy-Item system.env.example system.env
```

### 値を編集する

`system.env` をテキストエディタで開いて編集します。

**Mac（テキストエディット）：**

```bash
open -e system.env
```

**Windows（メモ帳）：**

```powershell
notepad system.env
```

**編集内容：**

```
# 評価用Judgeモデル（採点に使うAI）
JUDGE_MODEL=gpt-4o

# OpenAI APIキー（https://platform.openai.com/api-keys で取得）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Anthropic APIキー（https://console.anthropic.com/ で取得）
# OpenAIだけ使う場合は空欄のままでOK
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

> **APIキーの取得場所**
> - OpenAI: https://platform.openai.com/api-keys
> - Anthropic: https://console.anthropic.com/ → API Keys

---

## STEP 5 — テスト対象のAIモデルを登録する

`config/models.yaml` を開き、テストしたいAIのエンドポイントを追記します。

```yaml
models:
  # ---- 自分で立てたFastAPIサーバー ----
  - id: my_ai
    name: "社内AI v1"
    type: openai_compatible          # OpenAI互換エンドポイント
    base_url: "http://my-ai-server:8001/v1"
    api_key: "dummy"                 # 認証なしなら何でもOK
    default_params:
      temperature: 0.7
      max_tokens: 1024

  # ---- GPT-4o（比較用） ----
  - id: gpt4o
    name: "GPT-4o"
    type: openai
    model: "gpt-4o"
    api_key: "${OPENAI_API_KEY}"     # system.env の値が自動で入る
    default_params:
      temperature: 0.7
      max_tokens: 1024
```

テスト対象AIが同じDockerネットワーク上にある場合は
`docker-compose.yml` の末尾コメントを参考に `services:` へ追加してください。

---

## STEP 6 — 実行するテストを選ぶ

`config/settings.yaml` を開き、実行したいテストの `enabled:` を `true` にします。

```yaml
test_suites:
  mpi:
    enabled: true      # Big Five 性格診断（120問）
    sample_size: 20    # まず20問でお試し。null にすると全問実行

  japanese_rp_bench:
    enabled: false     # ロールプレイ評価（後で有効化）
    sample_size: 10
```

---

## STEP 7 — テストデータを用意する

`datasets/mpi/questions_ja.json` を作成します（最小サンプル）。
本番では後述の `import-github` コマンドで自動取得できます。

```json
[
  {"id": "q1", "text": "話し好きだ", "factor": "E", "keyed": "plus"},
  {"id": "q2", "text": "人に欠点を見つけがちだ", "factor": "A", "keyed": "minus"},
  {"id": "q3", "text": "仕事をしっかりとやり遂げる", "factor": "C", "keyed": "plus"},
  {"id": "q4", "text": "憂鬱になることがある", "factor": "N", "keyed": "plus"},
  {"id": "q5", "text": "独創的で、新しいアイデアを出す", "factor": "O", "keyed": "plus"}
]
```

---

## STEP 8 — コンテナをビルド＆起動する

```bash
# ビルドして起動（初回は数分かかります）
docker compose up --build
```

ログに `Application startup complete` と出れば起動成功です。

起動確認後、別のターミナルウィンドウを開いてテストを実行します。

---

## STEP 9 — テストを実行する

```bash
# 全モデル × 全テストを実行
docker compose exec eval-runner python main.py run

# 特定のモデルだけ実行
docker compose exec eval-runner python main.py run --models my_ai

# 特定のモデルを複数指定
docker compose exec eval-runner python main.py run --models my_ai gpt4o
```

実行が完了すると `reports/` フォルダにレポートが生成されます：

```
reports/
├── report_20250329_153000.html    ← ブラウザで開いて確認
└── report_20250329_153000.json    ← プログラムから読む場合
```

---

## STEP 10 — テストセットを追加する

### GitHubリポジトリから追加

```bash
# Japanese-RP-Bench を取り込む例
docker compose exec eval-runner python main.py import-github \
    https://github.com/Aratako/Japanese-RP-Bench \
    --name japanese_rp_bench \
    --pattern "*.json"
```

### arXiv論文から関連リポジトリを探す

```bash
# AIPsychoBench（arxiv: 2509.16530）の場合
docker compose exec eval-runner python main.py import-arxiv 2509.16530
```

表示されたGitHubリンクを確認し、上記 `import-github` コマンドで取り込みます。

---

## よくあるエラーと対処

| エラー | 原因 | 対処 |
|--------|------|------|
| `Cannot connect to the Docker daemon` | Docker Desktop が起動していない | Docker Desktop を起動する |
| `FileNotFoundError: system.env` | system.env がない | `cp system.env.example system.env` を実行 |
| `KeyError: 'OPENAI_API_KEY'` | system.env にキーが未設定 | system.env を編集してAPIキーを記入 |
| `Connection refused` (テスト対象AI) | AIサーバーが起動していない | AIサーバーを先に起動し、docker-compose.yml のURLを確認 |
| `port is already allocated` | 8080番ポートが使用中 | docker-compose.yml の `8080:8080` を `8081:8080` に変更 |

---

## コンテナの停止・再起動

```bash
# 停止（データは保持）
docker compose down

# 停止してボリュームも削除（完全リセット）
docker compose down -v

# コードを変更した場合は再ビルド
docker compose up --build
```

---

## ファイル構成まとめ

```
DigiMLab/
├── system.env            ★ APIキー設定（Gitに含めない）
├── system.env.example    　テンプレート（Gitに含めてOK）
├── config/
│   ├── models.yaml       ★ テスト対象AIのURL・設定
│   └── settings.yaml     ★ 実行するテストの選択
├── datasets/             　テストデータ置き場
├── evaluators/           　評価ロジック（拡張はここに追加）
├── reports/              　生成されたレポート出力先
└── docker-compose.yml    　コンテナ定義
```

★マークのファイルが日常的に編集するファイルです。
