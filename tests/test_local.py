"""不联网的本地测试：验证工具安全边界和基础行为。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.rag_tools import cosine_similarity, split_text
from app.workspace_tools import (
    list_workspace_files,
    read_text_file,
    resolve_workspace_path,
    save_new_text_file,
    search_workspace_text,
)


class WorkspaceToolTests(unittest.TestCase):
    def test_list_can_see_welcome_file(self) -> None:
        result = list_workspace_files.invoke({"relative_directory": "."})
        self.assertIn("welcome.md", result)

    def test_read_gets_real_project_code(self) -> None:
        result = read_text_file.invoke({"relative_path": "welcome.md"})
        self.assertIn("COMMON-CORE-01", result)

    def test_search_returns_file_and_line_number(self) -> None:
        result = search_workspace_text.invoke(
            {"query": "项目代号", "file_pattern": "*.md"}
        )
        self.assertIn("welcome.md:", result)

    def test_parent_path_escape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_workspace_path("../../.env")

    def test_save_really_writes_but_refuses_overwrite(self) -> None:
        # TemporaryDirectory 会建立一次性测试目录；测试结束后由 Python 自动清理。
        # patch 只在这个 with 代码块内，把工具的 workspace 临时换成测试目录，
        # 因此不会往大家真正学习用的 workspace 塞测试垃圾文件。
        with TemporaryDirectory() as temporary_directory:
            temporary_workspace = Path(temporary_directory).resolve()
            with patch("app.workspace_tools.WORKSPACE_ROOT", temporary_workspace):
                result = save_new_text_file.invoke(
                    {"relative_path": "notes/test.md", "content": "真实写入"}
                )
                self.assertIn("已真实新建文件", result)
                self.assertEqual(
                    (temporary_workspace / "notes/test.md").read_text(encoding="utf-8"),
                    "真实写入",
                )

                # 再写同一个文件必须失败，这证明“禁止覆盖”不是文档口号。
                with self.assertRaises(ValueError):
                    save_new_text_file.invoke(
                        {"relative_path": "notes/test.md", "content": "试图覆盖"}
                    )


class RagMathTests(unittest.TestCase):
    def test_short_text_stays_in_one_chunk(self) -> None:
        self.assertEqual(split_text("一小段知识"), ["一小段知识"])

    def test_long_text_is_split_with_overlap(self) -> None:
        chunks = split_text("A" * 1500)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(chunk) for chunk in chunks), 800)

    def test_cosine_similarity_distinguishes_direction(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
