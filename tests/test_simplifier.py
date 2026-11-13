import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ast_simplifier import JavaCodeSimplifier


class TestJavaCodeSimplifier(unittest.TestCase):
    def test_documented_example(self):
        source = "public static int square(int value) { return value * value; }"
        self.assertEqual(
            JavaCodeSimplifier().simplify(source, 0.5),
            "int square int value return value * value",
        )

    def test_ratio_boundaries_and_stable_order(self):
        simplifier = JavaCodeSimplifier()
        # All four identifiers have equal priority. Earlier tokens are removed first.
        source = "alpha beta gamma delta"
        for ratio, expected in [(0, source), (0.5, "gamma delta"), (1, "")]:
            with self.subTest(ratio=ratio):
                self.assertEqual(simplifier.simplify(source, ratio), expected)

    def test_cli_reads_java_file_and_rejects_nan(self):
        script = Path(__file__).resolve().parents[1] / "ast_simplifier.py"
        with TemporaryDirectory() as directory:
            file = Path(directory) / "Example.java"
            file.write_text("int answer() { return 42; }", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), str(file), "--ratio", "0"],
                capture_output=True, text=True, check=True,
            )
        self.assertEqual(result.stdout.splitlines()[-1], "int answer ( ) { return 42 ; }")
        result = subprocess.run(
            [sys.executable, str(script), "--ratio", "nan"], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--ratio", result.stderr)
