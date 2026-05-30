import argparse
import os
import sys
import textwrap

import requests

DEFAULT_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an expert DevOps engineer and AI workflow analyst.
    Analyze the following CI/CD workflow log and deliver a concise, actionable report.

    Your response should include these sections:
    1) Root Cause
    2) Failed Component
    3) Suggested Fix
    4) Severity
    5) Summary

    Keep responses brief, precise, and written for a DevOps engineer.

    Log:
    {log}
    """
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Workflow Analyzer using local Ollama + Llama3"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the CI/CD workflow log or failure output file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name to use (default: llama3).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Local Ollama HTTP API endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output file path to save the AI analysis.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum token length for the AI response.",
    )
    return parser.parse_args()


def read_log_file(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def build_prompt(log_text: str) -> str:
    return PROMPT_TEMPLATE.format(log=log_text)


def call_ollama(prompt: str, model: str, api_url: str, max_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Failed to reach Ollama API at {api_url}. Is the server running?\n{exc}"
        )

    data = response.json()
    if not isinstance(data, dict) or "choices" not in data:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")

    choice = data["choices"][0]
    return (
        choice.get("text")
        or choice.get("message", {}).get("content")
        or str(choice)
    )


def save_output(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> int:
    args = parse_args()

    try:
        log_text = read_log_file(args.input)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    prompt = build_prompt(log_text)

    print("[INFO] Sending log to Ollama for analysis...")
    try:
        result = call_ollama(prompt, args.model, args.api_url, args.max_tokens)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("\n=== AI Workflow Analysis ===\n")
    print(result)

    if args.output:
        save_output(args.output, result)
        print(f"\n[INFO] Saved analysis to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
