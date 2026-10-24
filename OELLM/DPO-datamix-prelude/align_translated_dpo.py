#!/usr/bin/env python3
"""Rewrite translated Dolci DPO JSONL files into the English parquet schema.

English stores prompt, chosen, and rejected as lists of messages. The
translations store chosen and rejected as plain assistant strings, and their
string columns use a different Arrow type. This script wraps those strings as
assistant messages and casts every column onto the English feature types.

Originals are left in place. One parquet file is written per language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

SOURCE = Path("/gpfs/projects/ehpc533/users/hannan/data/Dolci-Instruct-DPO-translated")
OUTPUT = Path("/gpfs/projects/ehpc533/users/hannan/data/Dolci-Instruct-DPO-translated-aligned")
LANGUAGES = ["cs", "de", "el", "es", "fi", "fr", "it", "pl", "ro", "sv", "uk"]


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _as_messages(value: object, default_role: str) -> list[dict[str, str]]:
    """Return role/content messages, wrapping a bare string as one message."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [{"role": default_role, "content": value}]
    messages: list[dict[str, str]] = []
    for message in value:
        content = _text(message.get("content"))
        if isinstance(message, str):
            if not message.strip():
                continue
            messages.append({"role": default_role, "content": message})
            continue
        messages.append({"role": _text(message.get("role")) or default_role, "content": content})
    return messages


def normalize(row: dict) -> dict:
    return {
        "id": _text(row.get("id")),
        "prompt": _as_messages(row.get("prompt"), "user"),
        "chosen": _as_messages(row.get("chosen"), "assistant"),
        "rejected": _as_messages(row.get("rejected"), "assistant"),
    }


def align_language(language: str, features, num_proc: int) -> None:
    source = SOURCE / language / "translated_dpo.jsonl"
    destination = OUTPUT / f"{language}.parquet"
    print(f"{language}: reading {source}", flush=True)
    dataset = load_dataset("json", data_files=str(source), split="train")
    dataset = dataset.map(
        normalize,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"normalizing {language}",
    )
    dataset = dataset.cast(features)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(destination)
    print(f"{language}: wrote {len(dataset)} rows to {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="*", default=LANGUAGES)
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    english = load_dataset(
        "parquet",
        data_dir=str(SOURCE / "en"),
        split="train[:1]",
    )
    features = english.features
    print(f"English schema: {features}", flush=True)

    for language in args.languages:
        align_language(language, features, args.num_proc)


if __name__ == "__main__":
    main()
