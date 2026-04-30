from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED_NOTEBOOK_DIR = ROOT / "docs" / "notebooks"
PUBLISHED_NOTEBOOKS: dict[Path, Path] = {
    ROOT / "src" / "notebooks" / "example_simulate_factory.py": (
        GENERATED_NOTEBOOK_DIR / "example_simulate_factory.ipynb"
    ),
    ROOT / "src" / "notebooks" / "example_simulate_factory_configurable.py": (
        GENERATED_NOTEBOOK_DIR / "example_simulate_factory_configurable.ipynb"
    ),
}


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def prepare_notebooks() -> None:
    GENERATED_NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    for notebook_path in GENERATED_NOTEBOOK_DIR.glob("*.ipynb"):
        notebook_path.unlink()

    for source_path, output_path in PUBLISHED_NOTEBOOKS.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Notebook source not found: {source_path}")

        _run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(source_path),
                "-o",
                str(output_path),
                "--include-outputs",
                "--force",
            ]
        )


def build_docs(strict: bool) -> None:
    prepare_notebooks()

    command = [sys.executable, "-m", "mkdocs", "build"]
    if strict:
        command.append("--strict")
    _run(command)


def serve_docs(addr: str) -> None:
    prepare_notebooks()
    _run([sys.executable, "-m", "mkdocs", "serve", "--dev-addr", addr])


def clean_docs() -> None:
    if GENERATED_NOTEBOOK_DIR.exists():
        for notebook_path in GENERATED_NOTEBOOK_DIR.glob("*.ipynb"):
            notebook_path.unlink()

    site_dir = ROOT / "site"
    if site_dir.exists():
        shutil.rmtree(site_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated notebook-driven documentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Export published marimo notebooks.")

    build_parser = subparsers.add_parser("build", help="Build the MkDocs site.")
    build_parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable MkDocs strict mode for local debugging.",
    )

    serve_parser = subparsers.add_parser("serve", help="Serve docs locally.")
    serve_parser.add_argument(
        "--dev-addr",
        default="127.0.0.1:8000",
        help="Address passed to `mkdocs serve --dev-addr`.",
    )

    subparsers.add_parser("clean", help="Remove generated docs artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "prepare":
        prepare_notebooks()
    elif args.command == "build":
        build_docs(strict=not args.no_strict)
    elif args.command == "serve":
        serve_docs(addr=args.dev_addr)
    elif args.command == "clean":
        clean_docs()


if __name__ == "__main__":
    main()
