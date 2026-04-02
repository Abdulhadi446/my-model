import argparse
import json
import re
from pathlib import Path


SKIP_USER_PREFIXES = (
    "Type @ to mention files",
    "shift+tab switch mode",
    "Remaining reqs.",
)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if set(stripped) <= set("─╭╮╯╰│▐▛███▜▌▝▜█████▛▘▘▝▔"):
        return True
    if stripped.startswith("claude-sonnet-"):
        return True
    if stripped.startswith("Claude Code v"):
        return True
    if stripped.startswith("kimi-"):
        return True
    return False


def is_valid_instruction(text: str, min_chars: int) -> bool:
    s = text.strip()
    if len(s) < min_chars:
        return False
    if any(s.startswith(prefix) for prefix in SKIP_USER_PREFIXES):
        return False
    return True


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    filtered_lines = []
    for line in text.split("\n"):
        if is_noise_line(line):
            continue
        filtered_lines.append(line.rstrip())
    text = "\n".join(filtered_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pairs(content: str):
    lines = content.splitlines()

    pairs = []
    current_user = None
    assistant_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # User prompts in these chat exports are typically prefixed with this glyph.
        if line.startswith("❯ "):
            if current_user is not None and assistant_lines:
                response = clean_text("\n".join(assistant_lines))
                if response:
                    pairs.append((clean_text(current_user), response))

            current_user = line[2:].strip()
            assistant_lines = []
            i += 1
            continue

        # Assistant sections are prefixed by a bullet marker.
        if line.startswith("● ") and current_user is not None:
            assistant_lines.append(line[2:])
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("❯ "):
                    break
                # Keep multiline assistant text, bash output, and wrapped paragraphs.
                assistant_lines.append(nxt)
                i += 1
            continue

        i += 1

    if current_user is not None and assistant_lines:
        response = clean_text("\n".join(assistant_lines))
        if response:
            pairs.append((clean_text(current_user), response))

    return pairs


def build_dataset(chats_dir: Path, out_path: Path, min_instruction_chars: int = 3):
    files = sorted(chats_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {chats_dir}")

    total_pairs = 0
    written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for f in files:
            content = f.read_text(encoding="utf-8", errors="ignore")
            pairs = extract_pairs(content)
            total_pairs += len(pairs)

            for instruction, response in pairs:
                if not is_valid_instruction(instruction, min_instruction_chars):
                    continue
                item = {
                    "instruction": instruction,
                    "response": response,
                    "source": str(f.name),
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    return {
        "files": len(files),
        "pairs_found": total_pairs,
        "rows_written": written,
        "output": str(out_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert chat logs to instruction/response JSONL for SFT"
    )
    parser.add_argument("--chats_dir", default="../chats")
    parser.add_argument("--output", default="data/train_from_chats.jsonl")
    parser.add_argument("--min_instruction_chars", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    chats_dir = Path(args.chats_dir).resolve()
    out_path = Path(args.output).resolve()

    stats = build_dataset(
        chats_dir=chats_dir,
        out_path=out_path,
        min_instruction_chars=args.min_instruction_chars,
    )

    print("Dataset build complete")
    print(f"- files processed: {stats['files']}")
    print(f"- pairs found: {stats['pairs_found']}")
    print(f"- rows written: {stats['rows_written']}")
    print(f"- output: {stats['output']}")


if __name__ == "__main__":
    main()
