"""Release tooling: tag/version validation and changelog extraction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import release_notes  # noqa: E402

CHANGELOG = """# Changelog

## [1.2.0] - 2026-11-01 - Next

### Added
- shiny

## [1.1.0] - 2026-10-18 - Production hardening release

### Fixed
- bug one
- bug two

## [1.0.0] - old
- ancient
"""


class ReleaseNotesTests(unittest.TestCase):
    def test_extracts_only_the_requested_section(self) -> None:
        notes = release_notes.extract_section(CHANGELOG, "1.1.0")
        self.assertIn("bug one", notes)
        self.assertNotIn("shiny", notes)
        self.assertNotIn("ancient", notes)
        self.assertTrue(notes.startswith("### Fixed"))

    def test_last_section_and_missing_version(self) -> None:
        self.assertIn("ancient", release_notes.extract_section(CHANGELOG, "1.0.0"))
        self.assertIsNone(release_notes.extract_section(CHANGELOG, "9.9.9"))
        self.assertIsNone(release_notes.extract_section(CHANGELOG, "1.1"))  # no prefix matching

    def test_cli_accepts_the_current_version_and_rejects_others(self) -> None:
        version = release_notes.read_version()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "notes.md"
            self.assertEqual(release_notes.main(["--tag", f"v{version}", "--out", str(out)]), 0)
            self.assertTrue(out.read_text(encoding="utf-8").strip())
        self.assertEqual(release_notes.main(["--tag", "v0.0.1"]), 1)
        self.assertEqual(release_notes.main(["--tag", version]), 1)  # missing the 'v' prefix

    def test_changelog_has_a_section_for_the_current_version(self) -> None:
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIsNotNone(release_notes.extract_section(text, release_notes.read_version()))


if __name__ == "__main__":
    unittest.main()
