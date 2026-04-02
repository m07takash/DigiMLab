"""
DigiMLab — バックグラウンドジョブランナー
Streamlit から subprocess で起動され、テストを実行して結果をファイルに書き出す。

使い方:
  python job_runner.py <job_id>
  ジョブ定義は JOBS_DIR/<job_id>/job.json から読む。
  進捗・結果は同ディレクトリの status.json, results.json に書き出す。

job_type:
  "mpi"       — MPI (Big Five) 採点付き実行
  "rp"        — 日本語ロールプレイ回答収集
  "rp_judge"  — 既存RP結果に対するLLM-as-Judge採点
  "custom_qa" — 任意の質問リスト（質問→回答のみ）
"""
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

JOBS_DIR = Path(__file__).parent / "reports" / "jobs"


def load_job(job_id: str) -> dict:
    job_file = JOBS_DIR / job_id / "job.json"
    return json.loads(job_file.read_text(encoding="utf-8"))


def write_status(job_id: str, status: dict):
    path = JOBS_DIR / job_id / "status.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results(job_id: str, results: list[dict]):
    path = JOBS_DIR / job_id / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


async def call_api(base_url: str, user_input: str, session_id: str, params: dict) -> dict:
    body = {
        "service_info": params.get("service_info", {"SERVICE_ID": "DigiMLab", "SERVICE_DATA": {}}),
        "user_info": params.get("user_info", {"USER_ID": "anonymous", "USER_DATA": {}}),
        "session_id": session_id,
        "user_input": user_input,
    }
    # 基本パラメータ（必ず送る）
    for key in ("agent_file", "engine", "situation"):
        if key in params:
            body[key] = params[key]
    # boolパラメータ（設定されていれば送る）
    for key in ("stream_mode", "save_digest", "memory_use",
                "magic_word_use", "meta_search", "rag_query_gene"):
        if key in params and params[key] is not None:
            body[key] = params[key]
    # web_search関連（web_searchがtrueの場合のみweb_search_engineを送る）
    if params.get("web_search"):
        body["web_search"] = True
        if "web_search_engine" in params:
            body["web_search_engine"] = params["web_search_engine"]
    elif "web_search" in params:
        body["web_search"] = False

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(base_url, json=body, headers={"Content-Type": "application/json"}) as resp:
            if resp.status != 200:
                # エラー詳細を記録
                try:
                    err_body = await resp.text()
                except Exception:
                    err_body = ""
                return {"error": f"HTTP {resp.status}: {err_body[:200]}", "response": ""}
            return await resp.json()


def _update_progress(job_id: str, job: dict, total: int, i: int, q: dict, results: list):
    q_label = q.get("text", q.get("question", ""))
    write_status(job_id, {
        "state": "running",
        "total": total,
        "completed": i,
        "errors": sum(1 for r in results if r.get("is_error")),
        "current_question": f"[{i+1}/{total}] {q.get('id', i+1)}: {q_label}",
        "started_at": job.get("_started_at", datetime.now().isoformat()),
        "finished_at": None,
    })


# ---------------------------------------------------------------------------
# MPI (Big Five) 実行
# ---------------------------------------------------------------------------

def _build_mpi_prompt(system_prompt: str, q: dict) -> str:
    return (
        f"{system_prompt}\n\n"
        f"次の文章について、あなた自身にどの程度当てはまりますか？"
        f"「{q['text']}」\n"
        f"1=全くそう思わない, 2=そう思わない, 3=どちらでもない, "
        f"4=そう思う, 5=とてもそう思う\n"
        f"数字1つだけで答えてください。"
    )


def _score_mpi(q: dict, raw_response: str) -> dict:
    m = re.search(r"[1-5]", raw_response)
    if not m:
        return {"rating": None, "adjusted": None, "normalized": None, "is_error": True}
    rating = int(m.group())
    adjusted = (6 - rating) if q.get("keyed") == "minus" else rating
    return {
        "rating": rating,
        "adjusted": adjusted,
        "normalized": adjusted / 5.0,
        "is_error": False,
    }


def _build_mpi_result(q: dict, raw_response: str, is_error: bool, scores: dict) -> dict:
    return {
        "id": q["id"],
        "text": q["text"],
        "factor": q.get("factor", ""),
        "keyed": q.get("keyed", ""),
        "raw_response": raw_response,
        "rating": scores.get("rating"),
        "adjusted": scores.get("adjusted"),
        "normalized": scores.get("normalized"),
        "is_error": is_error,
    }


# ---------------------------------------------------------------------------
# RP Bench（日本語ロールプレイ）実行
# ---------------------------------------------------------------------------

def _build_rp_prompt(system_prompt_template: str, q: dict) -> str:
    """system_promptテンプレート内の{persona},{persona_description}を展開し、user_inputを返す"""
    prompt = system_prompt_template.replace("{persona}", q.get("persona", ""))
    prompt = prompt.replace("{persona_description}", q.get("persona_description", ""))
    return prompt.strip() + "\n\n" + q.get("user_input", "")


def _build_rp_result(q: dict, raw_response: str, is_error: bool) -> dict:
    return {
        "id": q.get("id", ""),
        "persona": q.get("persona", ""),
        "persona_description": q.get("persona_description", ""),
        "user_input": q.get("user_input", ""),
        "reference": q.get("reference", ""),
        "raw_response": raw_response,
        "is_error": is_error,
    }


# ---------------------------------------------------------------------------
# RP Judge（LLM-as-Judge 採点）
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
あなたはロールプレイ応答の品質を評価する審査員です。
以下の評価基準に基づき、厳密かつ公平に採点してください。

■ 評価基準（各1〜5点）
- consistency: 一人称・語尾・口調がペルソナ設定と一貫しているか
  1=全く一致しない 2=一部一致 3=概ね一致するが揺れがある 4=ほぼ一致 5=完全に一致
- naturalness: 日本語として自然で読みやすいか
  1=不自然・破綻 2=違和感が多い 3=概ね自然 4=自然 5=非常に自然
- relevance: ユーザーの発話に対して的確に応じているか
  1=無関係 2=ずれている 3=概ね対応 4=的確 5=非常に的確
- persona_accuracy: ペルソナの背景・知識・価値観が応答に反映されているか
  1=全く反映なし 2=わずかに反映 3=部分的に反映 4=よく反映 5=非常によく反映

■ 評価対象

【ペルソナ設定】
{persona}: {persona_description}

【ユーザーの発話】
{user_input}

【AIの応答】
{response}

■ 出力形式
以下のJSON形式のみを出力してください。それ以外のテキストは不要です。
各観点のスコア（整数1〜5）と、その根拠を1〜2文で簡潔に記述してください。

{{"consistency": 4, "naturalness": 5, "relevance": 4, "persona_accuracy": 3, "reason": "一人称「あっし」や語尾「〜でぇ」は概ね再現されているが、一部現代的な表現が混在。発話への対応は的確で自然な日本語だが、魚屋としての専門知識の反映がやや弱い。"}}
"""


async def call_openai_judge(api_key: str, model: str, prompt: str) -> str:
    """OpenAI API を直接呼んでJudge採点する"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=body, headers=headers,
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"Judge API error HTTP {resp.status}: {err[:200]}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()


def run_rp_judge(job_id: str):
    """既存のRP結果に対してJudge採点を実行する"""
    job = load_job(job_id)
    source_job_id = job["source_job_id"]
    judge_config = job.get("judge_config", {})
    api_key = judge_config.get("api_key", "")
    model = judge_config.get("model", "gpt-5-mini")
    delay_seconds = job.get("delay_seconds", 1.0)
    max_retries = job.get("max_retries", 3)

    # 元ジョブの結果を読み込む
    source_results_path = JOBS_DIR / source_job_id / "results.json"
    results = json.loads(source_results_path.read_text(encoding="utf-8"))

    # エラーでない回答のみ採点対象
    targets = [(i, r) for i, r in enumerate(results) if not r.get("is_error")]
    total = len(targets)

    write_status(job_id, {
        "state": "running",
        "total": total,
        "completed": 0,
        "errors": 0,
        "current_question": "",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    })

    error_count = 0
    for step, (idx, r) in enumerate(targets):
        write_status(job_id, {
            "state": "running",
            "total": total,
            "completed": step,
            "errors": error_count,
            "current_question": f"[{step+1}/{total}] {r.get('id', '')} — {r.get('persona', '')} を採点中",
            "started_at": job.get("_started_at", datetime.now().isoformat()),
            "finished_at": None,
        })

        prompt = JUDGE_PROMPT.format(
            persona=r.get("persona", ""),
            persona_description=r.get("persona_description", ""),
            user_input=r.get("user_input", ""),
            response=r.get("raw_response", ""),
        )

        judge_scores = None
        for attempt in range(1 + max_retries):
            try:
                judge_response = asyncio.run(call_openai_judge(api_key, model, prompt))
                match = re.search(r"\{.*?\}", judge_response, re.DOTALL)
                if match:
                    judge_scores = json.loads(match.group())
                    break
                else:
                    print(f"  [Judge] {r.get('id','')}: JSON not found in: {judge_response[:100]}")
                    if attempt < max_retries:
                        time.sleep(2 ** (attempt + 1))
            except Exception as e:
                print(f"  [Judge Error] {r.get('id','')}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** (attempt + 1))

        if judge_scores:
            axes = ["consistency", "naturalness", "relevance", "persona_accuracy"]
            valid_scores = [judge_scores[a] for a in axes if a in judge_scores]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            results[idx]["judge_scores"] = {a: judge_scores[a] for a in axes if a in judge_scores}
            results[idx]["judge_reason"] = judge_scores.get("reason", "")
            results[idx]["judge_avg"] = round(avg / 5.0, 4)
        else:
            results[idx]["judge_scores"] = None
            results[idx]["judge_reason"] = None
            results[idx]["judge_avg"] = None
            error_count += 1

        # 途中結果を元ジョブの results.json に書き戻す
        source_results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if step < total - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    # 完了
    write_status(job_id, {
        "state": "completed",
        "total": total,
        "completed": total,
        "errors": error_count,
        "current_question": "",
        "started_at": job.get("_started_at", datetime.now().isoformat()),
        "finished_at": datetime.now().isoformat(),
    })
    print(f"Judge job {job_id} completed: {total - error_count}/{total} scored")


# ---------------------------------------------------------------------------
# Style（執筆スタイル再現度）実行
# ---------------------------------------------------------------------------

def _build_style_prompt(system_prompt: str, q: dict) -> str:
    """過去の文章サンプルを提示して、その人物のスタイルで書かせる"""
    samples = q.get("writing_samples", [])
    person = q.get("person", "この人物")
    samples_text = "\n\n".join(f"--- サンプル{i+1} ---\n{s}" for i, s in enumerate(samples))

    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    prompt_parts.append(
        f"以下は「{person}」が過去に書いた文章のサンプルです。\n\n"
        f"{samples_text}\n\n"
        f"上記の文章のスタイル（語彙、語尾、文体、考え方の傾向）を忠実に再現して、"
        f"以下のお題に対する文章を「{person}」として書いてください。\n\n"
        f"お題: {q.get('prompt', '')}"
    )
    return "\n\n".join(prompt_parts)


def _build_style_result(q: dict, raw_response: str, is_error: bool) -> dict:
    return {
        "id": q.get("id", ""),
        "person": q.get("person", ""),
        "prompt": q.get("prompt", ""),
        "writing_samples": q.get("writing_samples", []),
        "reference": q.get("reference", ""),
        "raw_response": raw_response,
        "is_error": is_error,
    }


# ---------------------------------------------------------------------------
# Style Judge（LLM-as-Judge 採点）
# ---------------------------------------------------------------------------

STYLE_JUDGE_PROMPT = """\
以下の「お題に対する文章」が、「過去の文章サンプル」の著者のスタイルをどの程度再現できているか評価してください。

【著者】
{person}

【過去の文章サンプル】
{samples}

【お題】
{prompt}

【AIが書いた文章】
{response}

以下の観点でそれぞれ1〜5点で採点し、JSON形式で返してください：
- substance: 話題の選び方・主張の方向性が著者らしいか
- vocabulary: 語彙の選び方・専門用語の使い方が著者らしいか
- tone: 語尾・口調・文体が著者らしいか
- coherence: 文章全体の論理構成・展開が著者らしいか

例：{{"substance": 4, "vocabulary": 3, "tone": 5, "coherence": 4}}
JSON以外は出力しないでください。
"""


def run_style_judge(job_id: str):
    """既存のStyle結果に対してJudge採点を実行する"""
    job = load_job(job_id)
    source_job_id = job["source_job_id"]
    judge_config = job.get("judge_config", {})
    api_key = judge_config.get("api_key", "")
    model = judge_config.get("model", "gpt-5-mini")
    delay_seconds = job.get("delay_seconds", 1.0)
    max_retries = job.get("max_retries", 3)

    source_results_path = JOBS_DIR / source_job_id / "results.json"
    results = json.loads(source_results_path.read_text(encoding="utf-8"))

    targets = [(i, r) for i, r in enumerate(results) if not r.get("is_error")]
    total = len(targets)

    write_status(job_id, {
        "state": "running", "total": total, "completed": 0, "errors": 0,
        "current_question": "", "started_at": datetime.now().isoformat(), "finished_at": None,
    })

    error_count = 0
    for step, (idx, r) in enumerate(targets):
        write_status(job_id, {
            "state": "running", "total": total, "completed": step, "errors": error_count,
            "current_question": f"[{step+1}/{total}] {r.get('id', '')} — {r.get('person', '')} を採点中",
            "started_at": job.get("_started_at", datetime.now().isoformat()), "finished_at": None,
        })

        samples_text = "\n\n".join(f"--- サンプル{i+1} ---\n{s}" for i, s in enumerate(r.get("writing_samples", [])))
        prompt = STYLE_JUDGE_PROMPT.format(
            person=r.get("person", ""),
            samples=samples_text,
            prompt=r.get("prompt", ""),
            response=r.get("raw_response", ""),
        )

        judge_scores = None
        for attempt in range(1 + max_retries):
            try:
                judge_response = asyncio.run(call_openai_judge(api_key, model, prompt))
                match = re.search(r"\{.*?\}", judge_response, re.DOTALL)
                if match:
                    judge_scores = json.loads(match.group())
                    break
                else:
                    if attempt < max_retries:
                        time.sleep(2 ** (attempt + 1))
            except Exception as e:
                print(f"  [Style Judge Error] {r.get('id','')}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** (attempt + 1))

        if judge_scores:
            axes = ["substance", "vocabulary", "tone", "coherence"]
            valid_scores = [judge_scores[a] for a in axes if a in judge_scores]
            avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            results[idx]["judge_scores"] = judge_scores
            results[idx]["judge_avg"] = round(avg / 5.0, 4)
        else:
            results[idx]["judge_scores"] = None
            results[idx]["judge_avg"] = None
            error_count += 1

        source_results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        if step < total - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    write_status(job_id, {
        "state": "completed", "total": total, "completed": total, "errors": error_count,
        "current_question": "", "started_at": job.get("_started_at", datetime.now().isoformat()),
        "finished_at": datetime.now().isoformat(),
    })
    print(f"Style Judge job {job_id} completed: {total - error_count}/{total} scored")


# ---------------------------------------------------------------------------
# カスタムQ&A 実行
# ---------------------------------------------------------------------------

def _build_qa_prompt(system_prompt: str, q: dict) -> str:
    question_text = q.get("question", q.get("text", ""))
    if system_prompt:
        return f"{system_prompt}\n\n{question_text}"
    return question_text


def _build_qa_result(q: dict, raw_response: str, is_error: bool) -> dict:
    return {
        "id": q.get("id", ""),
        "question": q.get("question", q.get("text", "")),
        "category": q.get("category", ""),
        "raw_response": raw_response,
        "is_error": is_error,
    }


# ---------------------------------------------------------------------------
# メインジョブ実行
# ---------------------------------------------------------------------------

def run_job(job_id: str):
    job = load_job(job_id)
    job_type = job.get("job_type", "mpi")
    questions = job["questions"]
    model_config = job["model_config"]
    system_prompt = job.get("system_prompt", "")
    session_id = job["session_id"]
    delay_seconds = job.get("delay_seconds", 0.5)
    max_retries = job.get("max_retries", 3)
    base_url = model_config["base_url"]
    params = model_config.get("default_params", {})

    total = len(questions)
    results = []

    write_status(job_id, {
        "state": "running",
        "total": total,
        "completed": 0,
        "errors": 0,
        "current_question": "",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    })

    for i, q in enumerate(questions):
        _update_progress(job_id, job, total, i, q, results)

        # プロンプト構築
        if job_type == "mpi":
            prompt = _build_mpi_prompt(system_prompt, q)
        elif job_type == "rp":
            prompt = _build_rp_prompt(system_prompt, q)
        elif job_type == "style":
            prompt = _build_style_prompt(system_prompt, q)
        else:
            prompt = _build_qa_prompt(system_prompt, q)

        # API呼び出し（リトライ付き）
        raw_response = ""
        call_success = False
        for attempt in range(1 + max_retries):
            try:
                resp = asyncio.run(call_api(base_url, prompt, session_id, params))
                # エラーレスポンスの場合
                if resp.get("error"):
                    raw_response = resp["error"]
                    print(f"  [API Error] {q.get('id', i+1)}: {raw_response}")
                    if attempt < max_retries:
                        time.sleep(2 ** (attempt + 1))
                    continue
                raw_response = str(resp.get("response", ""))
                call_success = True
                break
            except Exception as e:
                raw_response = f"ERROR: {e}"
                print(f"  [Exception] {q.get('id', i+1)}: {raw_response}")
                if attempt < max_retries:
                    time.sleep(2 ** (attempt + 1))
                continue

        # 結果レコード構築
        if job_type == "mpi":
            if call_success:
                scores = _score_mpi(q, raw_response)
                result = _build_mpi_result(q, raw_response, scores["is_error"], scores)
            else:
                result = _build_mpi_result(q, raw_response, True,
                                           {"rating": None, "adjusted": None, "normalized": None})
        elif job_type == "rp":
            result = _build_rp_result(q, raw_response, not call_success)
        elif job_type == "style":
            result = _build_style_result(q, raw_response, not call_success)
        else:
            result = _build_qa_result(q, raw_response, not call_success)

        results.append(result)
        write_results(job_id, results)

        if i < total - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    # 完了
    error_count = sum(1 for r in results if r.get("is_error"))
    write_status(job_id, {
        "state": "completed",
        "total": total,
        "completed": total,
        "errors": error_count,
        "current_question": "",
        "started_at": job.get("_started_at", datetime.now().isoformat()),
        "finished_at": datetime.now().isoformat(),
    })
    write_results(job_id, results)
    print(f"Job {job_id} completed: {total - error_count}/{total} valid")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python job_runner.py <job_id>")
        sys.exit(1)
    job_id = sys.argv[1]
    try:
        job_data = load_job(job_id)
        if job_data.get("job_type") == "rp_judge":
            run_rp_judge(job_id)
        elif job_data.get("job_type") == "style_judge":
            run_style_judge(job_id)
        else:
            run_job(job_id)
    except Exception as e:
        write_status(job_id, {
            "state": "failed",
            "error": str(e),
            "finished_at": datetime.now().isoformat(),
        })
        raise
