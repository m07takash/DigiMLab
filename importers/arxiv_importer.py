"""arXiv論文からデータセットのリンクを抽出・インポートする"""
import re
import httpx
from pathlib import Path


class ArxivImporter:
    """
    使用例:
        importer = ArxivImporter()
        info = importer.fetch_paper_info("2509.16530")  # AIPsychoBench
        print(info)  # title, abstract, github_links
    """

    def __init__(self):
        self.client = httpx.Client(timeout=30)

    def fetch_paper_info(self, arxiv_id: str) -> dict:
        """arXiv IDから論文情報とGitHubリンクを取得する"""
        url = f"https://arxiv.org/abs/{arxiv_id}"
        resp = self.client.get(url)
        resp.raise_for_status()

        html = resp.text
        title_match = re.search(r'<h1 class="title mathjax"[^>]*>(.*?)</h1>', html, re.DOTALL)
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""

        github_links = list(set(re.findall(r"https://github\.com/[\w\-]+/[\w\-]+", html)))

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "url": url,
            "github_links": github_links,
        }

    def suggest_import_steps(self, arxiv_id: str) -> str:
        """論文に関連するGitHubリポジトリをインポートするための手順を表示"""
        info = self.fetch_paper_info(arxiv_id)
        lines = [
            f"論文: {info['title']}",
            f"URL: {info['url']}",
            "",
            "関連GitHubリポジトリ:",
        ]
        for link in info["github_links"]:
            lines.append(f"  - {link}")

        lines += [
            "",
            "インポート例:",
            "  from importers.github_importer import GitHubImporter",
            "  importer = GitHubImporter()",
            f"  importer.import_repo(",
            f"      repo_url='{info['github_links'][0] if info['github_links'] else 'YOUR_REPO_URL'}',",
            f"      dataset_name='new_benchmark',",
            f"      file_pattern='*.json',",
            f"  )",
        ]
        return "\n".join(lines)
