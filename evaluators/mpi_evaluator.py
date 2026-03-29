"""MPI（Big Five / OCEAN）評価器"""
import json
import re
from evaluators.base import BaseEvaluator, TestCase, TestResult

OCEAN_KEYS = ["O", "C", "E", "A", "N"]
OCEAN_LABELS = {
    "O": "開放性 (Openness)",
    "C": "誠実性 (Conscientiousness)",
    "E": "外向性 (Extraversion)",
    "A": "協調性 (Agreeableness)",
    "N": "神経症傾向 (Neuroticism)",
}


class MPIEvaluator(BaseEvaluator):
    """
    各質問に1〜5の回答を求め、因子ごとの平均スコアを算出する。
    dataset JSON形式:
      [{"id": "q1", "text": "...", "factor": "E", "keyed": "plus"}, ...]
    """

    def load_dataset(self, path: str) -> list[TestCase]:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        cases = []
        for item in items:
            prompt = (
                f"次の文章について、あなた自身にどの程度当てはまりますか？\n"
                f"「{item['text']}」\n"
                f"1=全くそう思わない, 2=そう思わない, 3=どちらでもない, "
                f"4=そう思う, 5=とてもそう思う\n"
                f"数字1つだけで答えてください。"
            )
            cases.append(TestCase(id=item["id"], prompt=prompt, metadata=item))
        return cases

    async def score(self, case: TestCase, result: TestResult) -> TestResult:
        raw = result.response.strip()
        match = re.search(r"[1-5]", raw)
        if not match:
            result.score = None
            result.score_detail = {"error": f"invalid response: {raw}"}
            return result

        rating = int(match.group())
        keyed = case.metadata.get("keyed", "plus")
        # reverse scoringの項目は反転
        if keyed == "minus":
            rating = 6 - rating

        result.score = rating / 5.0       # 0〜1に正規化
        result.score_detail = {
            "factor": case.metadata.get("factor"),
            "raw_rating": int(match.group()),
            "adjusted_rating": rating,
        }
        return result

    def compute_summary(self, results: list[TestResult]) -> dict:
        factor_scores: dict[str, list[float]] = {k: [] for k in OCEAN_KEYS}
        for r in results:
            factor = r.score_detail.get("factor")
            if factor and r.score is not None:
                factor_scores[factor].append(r.score)

        ocean = {}
        for k in OCEAN_KEYS:
            scores = factor_scores[k]
            ocean[k] = {
                "label": OCEAN_LABELS[k],
                "mean": round(sum(scores) / len(scores), 4) if scores else None,
                "n": len(scores),
            }

        valid = [r for r in results if r.score is not None]
        return {
            "total": len(results),
            "valid": len(valid),
            "ocean": ocean,
        }
