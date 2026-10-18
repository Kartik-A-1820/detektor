#!/usr/bin/env python
"""Validate a release tag and extract its release notes from CHANGELOG.md.

    python scripts/release_notes.py --tag v1.1.0 --out notes.md

Fails (exit 1) when the tag does not match ``api.__version__`` or the changelog has no
section for that version, so a mis-tagged release can never be published.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


def read_version() -> str:
    text = (ROOT / "api" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, flags=re.M)
    if not match:
        raise SystemExit("error: could not find __version__ in api/__init__.py")
    return match.group(1)


def extract_section(changelog: str, version: str) -> Optional[str]:
    """Return the body of the ``## [version]`` section (heading excluded), or None."""
    pattern = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", flags=re.M | re.S)
    match = pattern.search(changelog)
    return match.group(1).strip() if match else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="Git tag, e.g. v1.1.0")
    parser.add_argument("--out", type=Path, help="Write the notes here (default: stdout)")
    args = parser.parse_args(argv)

    version = read_version()
    expected = f"v{version}"
    if args.tag != expected:
        print(f"error: tag {args.tag!r} does not match api.__version__ ({expected!r})", file=sys.stderr)
        return 1
    notes = extract_section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    if not notes:
        print(f"error: CHANGELOG.md has no '## [{version}]' section", file=sys.stderr)
        return 1
    if args.out:
        args.out.write_text(notes + "\n", encoding="utf-8")
    else:
        print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
