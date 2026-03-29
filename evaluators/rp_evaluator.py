"""日本語ロールプレイ評価器（LLM-as-Judge）"""
import json
from evaluators.base import BaseEvaluator, TestCase, TestResult


JUDGE_PROMPT = """\
以下のロールプレイ応答を評価してください。

【ペルソナ設定】
{persona}

【ユーザーの発話】
{user_input}

【AIの応答】
{response}

以下の観点でそれぞれ1〜5点で採点し、JSON形式で返してください：
- consistency: 一人称・語尾・口調がペルソナと一致しているか
- naturalness: 日本語として自然か
- relevance: 発話に対して適切に応じているか
- persona_accuracy: ペルソナの背景・知識・価値観が反映されているか

例：{{"consistency": 4, "naturalness": 5, "relevance": 4, "persona_accuracy": 3}}
JSON以外は出力しないでください。
"""


class RPEvaluator(BaseEvaluator):
    """
    dataset JSON形式:
      [{"id": "tc1", "persona": "...", "persona_description": "...",
        "user_input": "...", "reference": "..."}, ...]
    """

    def load_dataset(self, path: str) -> list[TestCase]:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return [
            TestCase(
                id=item["id"],
                prompt=item["user_input"],
                metadata=item,
            )
            for item in items
        ]

    async def score(self, case: TestCase, result: TestResult) -> TestResult:
        if not self.judge:
            result.score = None
            result.score_detail = {"error": "No judge model configured"}
            return result

        judge_input = JUDGE_PROMPT.format(
            persona=case.metadata.get("persona", ""),
            user_input=case.prompt,
            response=result.response,
        )
        try:
            judge_response = await self.judge.chat(
                [{"role": "user", "content": judge_input}],
                temperature=0,
            )
            import re, json as json_mod
            match = re.search(r"\{.*?\}", judge_response, re.DOTALL)
            if match:
                scores = json_mod.loads(match.group())
                avg = sum(scores.values()) / len(scores)
                result.score = round(avg / 5.0, 4)
                result.score_detail = scores
            else:
                result.score = None
                result.score_detail = {"raw": judge_response}
        except Exception as e:
            result.score = None
            result.score_detail = {"error": str(e)}
        return result

    def compute_summary(self, results: list[TestResult]) -> dict:
        valid = [r for r in results if r.score is not None]
        if not valid:
            return {"total": len(results), "valid": 0}

        axes = ["consistency", "naturalness", "relevance", "persona_accuracy"]
        axis_avgs = {}
        for ax in axes:
            vals = [r.score_detail.get(ax) for r in valid if isinstance(r.score_detail.get(ax), (int, float))]
            axis_avgs[ax] = round(sum(vals) / len(vals), 3) if vals else None

        return {
            "total": len(results),
            "valid": len(valid),
            "avg_score": round(sum(r.score for r in valid) / len(valid), 4),
            "axes": axis_avgs,
        }
