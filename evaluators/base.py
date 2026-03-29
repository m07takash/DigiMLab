"""評価器の基底クラス"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import asyncio
from runners.api_client import AIClient


@dataclass
class TestCase:
    id: str
    prompt: str
    metadata: dict = field(default_factory=dict)
    expected: Any = None


@dataclass
class TestResult:
    case_id: str
    model_id: str
    response: str
    score: float | None = None
    score_detail: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalReport:
    suite_name: str
    model_id: str
    model_name: str
    results: list[TestResult]
    summary: dict = field(default_factory=dict)


class BaseEvaluator(ABC):
    def __init__(self, config: dict, judge_client: AIClient | None = None):
        self.config = config
        self.judge = judge_client

    @abstractmethod
    def load_dataset(self, path: str) -> list[TestCase]:
        """テストセットを読み込む"""
        ...

    @abstractmethod
    async def score(self, case: TestCase, result: TestResult) -> TestResult:
        """スコアリング。resultにscore/score_detailを設定して返す"""
        ...

    async def run(
        self,
        client: AIClient,
        dataset_path: str,
        sample_size: int | None = None,
        concurrency: int = 3,
    ) -> EvalReport:
        cases = self.load_dataset(dataset_path)
        if sample_size:
            cases = cases[:sample_size]

        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def run_one(case: TestCase) -> TestResult:
            async with semaphore:
                system_prompt = self.config.get("system_prompt", "")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": case.prompt})
                try:
                    response = await client.chat(messages)
                    result = TestResult(
                        case_id=case.id,
                        model_id=client.config.id,
                        response=response,
                    )
                    result = await self.score(case, result)
                except Exception as e:
                    result = TestResult(
                        case_id=case.id,
                        model_id=client.config.id,
                        response="",
                        error=str(e),
                    )
                return result

        tasks = [run_one(c) for c in cases]
        results = await asyncio.gather(*tasks)

        summary = self.compute_summary(list(results))
        return EvalReport(
            suite_name=self.config.get("description", ""),
            model_id=client.config.id,
            model_name=client.config.name,
            results=list(results),
            summary=summary,
        )

    def compute_summary(self, results: list[TestResult]) -> dict:
        valid = [r for r in results if r.score is not None]
        if not valid:
            return {"error": "No valid results"}
        avg = sum(r.score for r in valid) / len(valid)
        return {"total": len(results), "valid": len(valid), "avg_score": round(avg, 4)}
