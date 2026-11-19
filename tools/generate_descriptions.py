"""Generate class descriptions with an LLM using the prompt of Sec. 3.4.

Each call asks for ``--per-call`` descriptions of at most ``--max-words`` words for one class
and writes one JSON file ``{class name: [descriptions]}`` covering all 147 class names, named
``descriptions/<model>/call_k<per-call>_w<max-words>_<i>.json`` (e.g. ``call_k20_w3_1.json``).

    export OPENAI_API_KEY=...                       # or OPENROUTER_API_KEY + --base-url
    python tools/generate_descriptions.py --model gpt-5.2 --calls 5 --per-call 20 --max-words 3
    # -> descriptions/gpt-5.2/call_k20_w3_1.json ... call_k20_w3_5.json

Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, ...). Generation is a one-time step;
the files used in the paper are committed under ``descriptions/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from declip.classnames import all_classnames  # noqa: E402

PROMPT = """\
Act as a Dermatologist and AI Data Curator. Generate {n} distinct, medically accurate captions for class: "{classname}".

DISTRIBUTION STRATEGY:
- 25% Morphology (Shape, borders, symmetry, elevation)
- 15% Color (Medical terms like erythematous, violaceous)
- 15% Texture (Scaly, ulcerated, verrucous)
- 25% Dermoscopy (Vascular patterns, pigment networks, structures)
- 10% Pathophysiology (Underlying mechanism visible)
- 10% Anatomical Context (Location, arrangement)

FORMATTING RULES:
1. Output ONLY a plain text numbered list.
2. NO Markdown code blocks, NO quotes, NO Python syntax.
3. Total lines: exactly {n}.
4. MAX {max_words} WORDS PER CAPTION.

OUTPUT TEMPLATE
1. Caption 1...
2. Caption 2...
...
{n}. Caption {n}..."""

LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def parse_list(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        match = LINE.match(line)
        if match:
            items.append(match.group(1).strip().strip('"').rstrip("."))
    return items


def generate_class(client: OpenAI, model: str, classname: str, n: int, max_words: int) -> list[str]:
    prompt = PROMPT.format(n=n, classname=classname, max_words=max_words)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8096
    )
    items = parse_list(response.choices[0].message.content or "")
    if len(items) != n:
        print(f"warning: {classname}: {len(items)} descriptions instead of {n}", file=sys.stderr)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--calls", type=int, default=1, help="independent calls per class (one file each)"
    )
    parser.add_argument("--start", type=int, default=1, help="index of the first call")
    parser.add_argument("--per-call", type=int, default=20, help="descriptions requested per call")
    parser.add_argument("--max-words", type=int, default=3)
    parser.add_argument(
        "--out-dir", default=None, help="default: descriptions/<model> (last path part of --model)"
    )
    parser.add_argument(
        "--name-pattern",
        default="call_k{k}_w{w}_{i}.json",
        help="file name; {k} = --per-call, {w} = --max-words, {i} = call index",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    classnames = all_classnames()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "descriptions",
        args.model.rstrip("/").split("/")[-1],
    )
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.start, args.start + args.calls):
        name = args.name_pattern.format(k=args.per_call, w=args.max_words, i=i)
        out = os.path.join(out_dir, name)
        if os.path.exists(out):
            print(f"{out} exists, skipping")
            continue
        results: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_class, client, args.model, c, args.per_call, args.max_words): c
                for c in classnames
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"call {i}"):
                results[futures[future]] = future.result()
        empty = [c for c, v in results.items() if not v]
        if empty:
            raise RuntimeError(f"call {i}: no descriptions for {empty[:5]}")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({c: results[c] for c in classnames}, handle, ensure_ascii=False, indent=2)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
