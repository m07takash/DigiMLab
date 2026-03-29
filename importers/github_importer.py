"""GitHubリポジトリからテストセットをインポートする"""
import subprocess
import shutil
import json
from pathlib import Path


class GitHubImporter:
    """
    使用例:
        importer = GitHubImporter("datasets/imported")
        importer.import_repo(
            repo_url="https://github.com/Aratako/Japanese-RP-Bench",
            dataset_name="japanese_rp_bench",
            file_pattern="*.json",
            converter=my_converter_func,  # 省略可
        )
    """

    def __init__(self, output_dir: str = "datasets/imported"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def import_repo(
        self,
        repo_url: str,
        dataset_name: str,
        file_pattern: str = "*.json",
        converter=None,
    ) -> Path:
        tmp_dir = Path(f"/tmp/eval_import_{dataset_name}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        print(f"[GitHub] Cloning {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(tmp_dir)],
            check=True,
        )

        out_dir = self.output_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        matched = list(tmp_dir.rglob(file_pattern))
        print(f"[GitHub] Found {len(matched)} file(s) matching '{file_pattern}'")

        for src in matched:
            if converter:
                with open(src, encoding="utf-8") as f:
                    raw = json.load(f)
                converted = converter(raw)
                dst = out_dir / src.name
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(converted, f, ensure_ascii=False, indent=2)
            else:
                shutil.copy(src, out_dir / src.name)

        shutil.rmtree(tmp_dir)
        print(f"[GitHub] Saved to {out_dir}")
        return out_dir

    @staticmethod
    def generate_registry_entry(dataset_name: str, evaluator_class: str, description: str) -> dict:
        """config/settings.yaml に追加するエントリを生成する"""
        return {
            dataset_name: {
                "enabled": True,
                "description": description,
                "dataset": f"datasets/imported/{dataset_name}",
                "evaluator": evaluator_class,
                "sample_size": None,
            }
        }
