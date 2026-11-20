"""Generate kernel implementations for every (LLM, kernel_type) pair in config/llm_lineup.yaml,
using the prompts under prompts/, and save them into kernels/<kernel>/llm_<name>/.

Requires API keys as environment variables (copy .env.example to .env and fill in, or export
them in your shell -- never paste real keys into chat/logs):
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, DEEPSEEK_API_KEY

Usage:
    python scripts/generate_kernels.py --dry-run                 # assemble + print prompts, no API calls, no cost
    python scripts/generate_kernels.py                            # all 5 LLMs x all 5 kernels
    python scripts/generate_kernels.py --llm claude --kernel matmul
    python scripts/generate_kernels.py --overwrite                # regenerate even if v1.py already exists
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
KERNELS_DIR = ROOT / "kernels"
CONFIG_PATH = ROOT / "config" / "llm_lineup.yaml"

KERNEL_TYPES = ["reduction", "softmax", "matmul", "conv2d", "attention"]


def load_dotenv_if_present():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def extract_fenced_block(markdown_text: str) -> str:
    """Pull the content of the first ``` ... ``` fenced block out of a markdown string."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", markdown_text, re.DOTALL)
    if not match:
        raise ValueError("No fenced code block found")
    return match.group(1).strip()


def build_prompt(kernel_type: str) -> tuple[str, str]:
    """Returns (prompt_id, full_prompt_text) for a kernel type."""
    header_text = (PROMPTS_DIR / "common_header.md").read_text()
    preamble = extract_fenced_block(header_text)

    body_text = (PROMPTS_DIR / f"{kernel_type}_prompt_v1.md").read_text()
    lines = body_text.splitlines()
    prompt_id = lines[0].split(":", 1)[1].strip()
    # Drop the "prompt_id: ..." line and the "(Prepend common_header.md's preamble...)" line --
    # both are authoring metadata, not part of what the LLM should see.
    body_lines = [
        ln for ln in lines[1:]
        if not ln.strip().startswith("(Prepend")
    ]
    body = "\n".join(body_lines).strip()

    full_prompt = f"{preamble}\n\n{body}"
    return prompt_id, full_prompt


def extract_generated_code(response_text: str) -> str:
    """LLMs sometimes wrap code in a fence, sometimes (despite instructions) add prose around
    it. Try to pull the fenced block; fall back to the raw text if there is none."""
    try:
        return extract_fenced_block(response_text)
    except ValueError:
        return response_text.strip()


# --- Provider call functions -------------------------------------------------------------

def call_claude(model_id: str, prompt: str, temperature: float) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model_id,
        max_tokens=4096,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def call_openai_compatible(model_id: str, prompt: str, temperature: float,
                            base_url: str = None, api_key_env: str = "OPENAI_API_KEY") -> str:
    from openai import OpenAI
    kwargs = {"api_key": os.environ[api_key_env]}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model_id,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_gemini(model_id: str, prompt: str, temperature: float) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client()
    resp = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return resp.text


PROVIDER_DISPATCH = {
    "Anthropic": lambda model_id, prompt, temp: call_claude(model_id, prompt, temp),
    "OpenAI": lambda model_id, prompt, temp: call_openai_compatible(
        model_id, prompt, temp, api_key_env="OPENAI_API_KEY"),
    "Google": lambda model_id, prompt, temp: call_gemini(model_id, prompt, temp),
    "xAI": lambda model_id, prompt, temp: call_openai_compatible(
        model_id, prompt, temp, base_url="https://api.x.ai/v1", api_key_env="XAI_API_KEY"),
    "DeepSeek": lambda model_id, prompt, temp: call_openai_compatible(
        model_id, prompt, temp, base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY"),
}


def generate_one(llm: dict, kernel_type: str, temperature: float, dry_run: bool,
                  overwrite: bool, attempt: int = 1):
    out_dir = KERNELS_DIR / kernel_type / f"llm_{llm['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / "v1.py"
    meta_path = out_dir / "v1.meta.json"
    raw_path = out_dir / "v1.raw.md"  # full response kept for audit, even if extraction succeeds

    if code_path.exists() and not overwrite:
        print(f"  skip {llm['name']:10s} {kernel_type:10s} (v1.py exists, use --overwrite to redo)")
        return

    prompt_id, prompt = build_prompt(kernel_type)

    if dry_run:
        print(f"\n--- {llm['name']} / {kernel_type} (prompt_id={prompt_id}) ---")
        print(prompt)
        return

    if not llm.get("model_id"):
        print(f"  SKIP {llm['name']:10s} {kernel_type:10s} -- model_id not set in "
              f"config/llm_lineup.yaml")
        return

    dispatch = PROVIDER_DISPATCH.get(llm["lab"])
    if dispatch is None:
        print(f"  SKIP {llm['name']:10s} {kernel_type:10s} -- no provider wired up for "
              f"lab '{llm['lab']}'")
        return

    try:
        response_text = dispatch(llm["model_id"], prompt, temperature)
    except Exception as e:  # noqa: BLE001 -- a failed generation call is a logged result, not a crash
        print(f"  ERROR {llm['name']:10s} {kernel_type:10s} -- {e}")
        meta_path.write_text(json.dumps({
            "llm_name": llm["name"], "lab": llm["lab"], "model_id": llm["model_id"],
            "gen_temperature": temperature, "prompt_id": prompt_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt,
            "error": str(e),
        }, indent=2))
        return

    raw_path.write_text(response_text)
    code = extract_generated_code(response_text)
    code_path.write_text(code)
    meta_path.write_text(json.dumps({
        "llm_name": llm["name"],
        "lab": llm["lab"],
        "model_id": llm["model_id"],
        "gen_temperature": temperature,
        "prompt_id": prompt_id,
        "prompt_quality_score": None,  # fill in by hand using prompts/rubric.md
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "attempt": attempt,
        "notes": "",
    }, indent=2))
    print(f"  OK   {llm['name']:10s} {kernel_type:10s} -> {code_path.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default=None, help="restrict to one LLM name from llm_lineup.yaml")
    p.add_argument("--kernel", default=None, choices=KERNEL_TYPES)
    p.add_argument("--dry-run", action="store_true", help="print assembled prompts, make no API calls")
    p.add_argument("--overwrite", action="store_true", help="regenerate even if v1.py already exists")
    args = p.parse_args()

    load_dotenv_if_present()
    lineup = yaml.safe_load(CONFIG_PATH.read_text())
    llms = lineup["llms"]
    temperature = lineup.get("default_temperature", 0.7)

    if args.llm:
        llms = [llm for llm in llms if llm["name"] == args.llm]
        if not llms:
            raise SystemExit(f"No LLM named '{args.llm}' in {CONFIG_PATH}")
    kernel_types = [args.kernel] if args.kernel else KERNEL_TYPES

    for kernel_type in kernel_types:
        print(f"\n=== {kernel_type} ===")
        for llm in llms:
            generate_one(llm, kernel_type, temperature, args.dry_run, args.overwrite)


if __name__ == "__main__":
    main()
