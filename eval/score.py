"""Scores the answers collected by run.py.

For every answer it checks the five coding rules in rules.md, compiles the
code and runs its tests with cargo, and flags "lazy" behaviour:

    false_claim    the prose says the rules were followed, but the code breaks one
    placeholder    the code elides parts ("// ... rest unchanged")
    dropped_tests  turn 2 lost the unit tests that turn 1 had
    dropped_error  turn 2 lost the custom error enum that turn 1 had

    python3 eval/score.py            # writes results/scores.csv and prints summary tables
"""

import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CRATE = RESULTS / ".crate"
MODELS = {"trtllm": "Qwen2.5-Coder-7B (TensorRT-LLM, GPTQ-Int4)", "llamacpp": "Qwen3.5-9B (llama.cpp, Q4_K_XL)"}
STD_ROOTS = {"std", "core", "alloc", "crate", "super", "self"}
# Paths like `fs::write` or `u8::MAX` start with a std module or a primitive, not a crate.
NOT_CRATES = STD_ROOTS | set(
    "alloc any array ascii borrow boxed cell char clone cmp collections convert default env error ffi fmt fs "
    "future hash hint io iter marker mem net num ops option os panic path pin prelude process ptr rc result "
    "slice str string sync task thread time vec bool f32 f64 i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 "
    "usize".split()
)


# ---------------------------------------------------------------- extraction


def screen_blocks(screen):
    """Code blocks drawn in the app's transcript: rows behind the "▏ " gutter, one block per header."""
    blocks, current = [], None
    for row in screen.splitlines():
        stripped = row.lstrip()
        if not stripped.startswith("▏"):
            current = None
            continue
        body = stripped[2:] if stripped.startswith("▏ ") else stripped[1:]
        if current is None or re.fullmatch(r"(\w+\s+)?copy\s*", body):
            current = []
            blocks.append(current)
            if re.fullmatch(r"(\w+\s+)?copy\s*", body):
                continue
        current.append(body)
    return ["\n".join(block) for block in blocks]


ITEM_START = re.compile(r"^(#\[|///|//!|use\s|pub\b|fn\s|struct\s|enum\s|impl\b|mod\s|trait\s|type\s|const\s|static\s|"
                        r"extern\s|unsafe\s|async\s)")
ITEM_KEY = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+|async\s+)*(fn|struct|enum|trait|mod|type|const|static)\s+(\w+)")


def top_level_items(block):
    """Splits a code block into top-level items (with their docs and attributes); loose statements are dropped."""
    items, pending, current, depth = [], [], None, 0
    for line in block.splitlines():
        code = strip_strings(line).split("//")[0]
        if current is None:
            if depth == 0 and not line.strip():
                pending = []
                continue
            if depth == 0 and ITEM_START.match(line):
                if line.lstrip().startswith(("#[", "///", "//!")) and not ITEM_KEY.match(line.split("]")[-1].strip()):
                    pending.append(line)
                    continue
                current = pending + [line]
                pending = []
            elif depth == 0:
                depth += code.count("{") - code.count("}")
                pending = []
                continue
            else:
                depth += code.count("{") - code.count("}")
                continue
        else:
            current.append(line)
        depth += code.count("{") - code.count("}")
        if depth <= 0 and ("{" in "".join(current) or code.rstrip().endswith(";")):
            depth = 0
            items.append("\n".join(current))
            current = None
    return items


def item_key(item):
    body = [l for l in item.splitlines() if not l.lstrip().startswith(("#[", "///", "//"))]
    head = body[0].strip() if body else item.strip()
    match = ITEM_KEY.match(head)
    if match:
        return match.group(1, 2)
    if head.startswith("use"):
        return ("use", head)
    return ("item", re.sub(r"\s+", " ", head.split("{")[0]))


def assemble(blocks):
    """One compilable unit from an answer's code blocks, as a reader would put it together: every
    top-level item once, a later version of an item replacing an earlier one, loose example
    statements dropped."""
    order, latest = [], {}
    for block in blocks:
        for item in top_level_items(block):
            key = item_key(item)
            if key not in latest:
                order.append(key)
            latest[key] = item
    return "\n\n".join(latest[key] for key in order)


def code_and_prose(record):
    """The Rust code of an answer (all its code blocks, assembled), and the prose around it."""
    if record["backend"] == "llamacpp":
        text = record["text"]
        blocks = [body for lang, body in re.findall(r"```(\w*)\n(.*?)```", text, flags=re.S) if lang in ("rust", "rs", "")]
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
    else:
        screen = record["screen"]
        blocks = screen_blocks(screen)
        prose = "\n".join(line for line in screen.splitlines() if not line.lstrip().startswith("▏"))
    return assemble(blocks), prose


# ---------------------------------------------------------------- rules


def strip_strings(code):
    """The code with string and char literal contents blanked, so '//' in a string is not a comment."""
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if c == '"':
            j = i + 1
            while j < n and code[j] != '"':
                j += 2 if code[j] == "\\" else 1
            out.append('""')
            i = j + 1
        elif c == "'" and i + 2 < n and (code[i + 2] == "'" or code[i + 1] == "\\"):
            j = code.find("'", i + 2 if code[i + 1] != "\\" else i + 3)
            out.append("' '")
            i = (j if j != -1 else i + 2) + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def comments_in_fn_bodies(code):
    """Number of comments inside function bodies (doc comments on items do not count)."""
    code = strip_strings(code)
    count, depth, bodies, pending_fn = 0, 0, [], False
    i = 0
    while i < len(code):
        if code.startswith("//", i):
            end = code.find("\n", i)
            end = len(code) if end == -1 else end
            if bodies and depth >= bodies[-1]:
                count += 1
            i = end
            continue
        if code.startswith("/*", i):
            end = code.find("*/", i)
            if bodies and depth >= bodies[-1]:
                count += 1
            i = len(code) if end == -1 else end + 2
            continue
        if re.match(r"\bfn\b", code[i : i + 3]) and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_")):
            pending_fn = True
        c = code[i]
        if c == "{":
            depth += 1
            if pending_fn:
                bodies.append(depth)
                pending_fn = False
        elif c == "}":
            if bodies and depth == bodies[-1]:
                bodies.pop()
            depth -= 1
        elif c == ";" and pending_fn:
            pending_fn = False  # a declaration without a body
        i += 1
    return count


def undocumented_pub_items(code):
    lines = code.splitlines()
    pub = re.compile(r"^\s*pub(\([^)]*\))?\s+(unsafe\s+)?(async\s+)?(fn|struct|enum|trait|type|const|static|mod)\b")
    total, missing = 0, 0
    for index, line in enumerate(lines):
        if not pub.match(line):
            continue
        total += 1
        above = index - 1
        while above >= 0 and lines[above].strip().startswith("#["):
            above -= 1
        if above < 0 or not lines[above].strip().startswith("///"):
            missing += 1
    return total, missing


def rules(code):
    no_strings = strip_strings(code)
    imported = set(re.findall(r"^\s*(?:pub\s+)?use\s+:{0,2}(\w+)::", no_strings, flags=re.M))
    # Crates used by full path without a `use`, e.g. `tempfile::NamedTempFile::new()`.
    local = set(re.findall(r"\b(?:mod|let|fn|as)\s+(?:mut\s+)?(\w+)", no_strings))
    local |= set(re.findall(r"\buse\s+[\w:]*::\{?([\w, ]+)\}?;", no_strings) and
                 ",".join(re.findall(r"\buse\s+[\w:]*::\{?([\w, ]+)\}?;", no_strings)).replace(" ", "").split(","))
    by_path = {root for root in re.findall(r"(?<![\w:.])([a-z_]\w*)::(?!<)", no_strings)} - local
    external = sorted((imported | by_path | set(re.findall(r"extern\s+crate\s+(\w+)", no_strings))) - NOT_CRATES)
    has_enum = bool(re.search(r"\benum\s+\w+", no_strings))
    # `use std::error::Error as StdError;` and the like: accept the alias too.
    display_names = ["Display"] + re.findall(r"fmt::Display\s+as\s+(\w+)", no_strings)
    error_names = ["Error"] + re.findall(r"error::Error\s+as\s+(\w+)", no_strings)
    has_display = bool(re.search(rf"impl\s*(<[^>]*>)?\s*(std::)?(fmt::)?({'|'.join(display_names)})\s+for", no_strings))
    has_error = bool(re.search(rf"impl\s*(<[^>]*>)?\s*(std::)?(error::)?({'|'.join(error_names)})\s+for", no_strings))
    pub_total, pub_missing = undocumented_pub_items(code)
    return {
        "r1_unwrap_expect": len(re.findall(r"\.(unwrap|expect)\s*\(", no_strings)),
        "r2_external_crates": ",".join(external),
        "r2_ok": has_enum and has_display and has_error and not external,
        "r3_pub_items": pub_total,
        "r3_undocumented": pub_missing,
        "r4_body_comments": comments_in_fn_bodies(code),
        "r5_index_loops": len(re.findall(r"\bfor\s+\w+\s+in\s+\(?\s*0\s*\.\.", no_strings)),
    }


# ---------------------------------------------------------------- compile and test


def cargo_test(code):
    (CRATE / "src").mkdir(parents=True, exist_ok=True)
    (CRATE / "Cargo.toml").write_text('[package]\nname = "answer"\nversion = "0.1.0"\nedition = "2021"\n')
    (CRATE / "src" / "lib.rs").write_text("#![allow(dead_code, unused)]\n" + code)
    build = subprocess.run(["cargo", "test", "--offline", "--no-run", "-q"], cwd=CRATE, capture_output=True, text=True)
    if build.returncode != 0:
        return False, False
    run = subprocess.run(["cargo", "test", "--offline", "-q"], cwd=CRATE, capture_output=True, text=True, timeout=120)
    return True, run.returncode == 0 and "0 passed" not in run.stdout


# ---------------------------------------------------------------- laziness

CLAIM = re.compile(
    r"follows? (all )?(the )?(provided |given |your )?(rules|guidelines|requirements)|adher|compl(y|ies|iant)|"
    r"without (using )?`?(unwrap|expect)|no `?(unwrap|expect)|avoid(s|ing)? `?(unwrap|expect)|no external crates",
    re.I,
)
PLACEHOLDER = re.compile(r"//\s*\.\.\.|/\*\s*\.\.\.|rest of (the )?(code|file|implementation)|remains? (the )?same|"
                         r"unchanged|same as (before|above)|previous code", re.I)


def score_record(record, turn1=None):
    code, prose = code_and_prose(record)
    result = {"backend": record["backend"], "temperature": record["temperature"], "run": record["run"],
              "turn": record["turn"], "tokens": record["tokens"], "tok_per_s": record["tok_per_s"],
              "prose_words": len(prose.split()), "code_lines": len(code.splitlines())}
    result.update(rules(code))
    violations = [
        result["r1_unwrap_expect"] > 0,
        not result["r2_ok"],
        result["r3_undocumented"] > 0,
        result["r4_body_comments"] > 0,
        result["r5_index_loops"] > 0,
    ]
    result["rules_broken"] = sum(violations)
    result["all_rules"] = not any(violations)
    result["compiles"], result["tests_pass"] = cargo_test(code)
    result["false_claim"] = bool(CLAIM.search(prose)) and any(violations)
    result["placeholder"] = bool(PLACEHOLDER.search(code))
    if turn1 is not None:
        result["dropped_tests"] = "#[test]" in turn1[0] and "#[test]" not in code
        result["dropped_error"] = bool(re.search(r"\benum\s+\w+", turn1[0])) and not re.search(r"\benum\s+\w+", code)
    return result, code


# ---------------------------------------------------------------- summary


def pct(values):
    values = [bool(v) for v in values]
    return f"{100 * sum(values) / len(values):.0f}%" if values else "n/a"


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["temperature"])].append(row)
    header = ("| Model | T | n | All 5 rules | R1 no unwrap | R2 error enum | R3 docs | R4 no body comments | "
              "R5 iterators | Has pub items | Compiles | Tests pass | False claim | Placeholder | Dropped (turn 2) | "
              "Prose words | tok/s |")
    print(header)
    print("|" + "|".join("---" for _ in range(header.count("|") - 1)) + "|")
    for backend in sorted({key[0] for key in groups}):
        groups[(backend, "all")] = [row for row in rows if row["backend"] == backend]
    for (backend, temperature), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        turn2 = [r for r in group if r["turn"] == 2]
        speeds = [r["tok_per_s"] for r in group if r["tok_per_s"]]
        speed = f"{sum(speeds) / len(speeds):.1f}" if speeds else "n/a"
        print(
            f"| {MODELS[backend]} | {temperature if temperature == 'all' else f'{temperature:g}'} | {len(group)} "
            f"| {pct(r['all_rules'] for r in group)} "
            f"| {pct(r['r1_unwrap_expect'] == 0 for r in group)} | {pct(r['r2_ok'] for r in group)} "
            f"| {pct(r['r3_undocumented'] == 0 for r in group)} | {pct(r['r4_body_comments'] == 0 for r in group)} "
            f"| {pct(r['r5_index_loops'] == 0 for r in group)} | {pct(r['r3_pub_items'] > 0 for r in group)} "
            f"| {pct(r['compiles'] for r in group)} "
            f"| {pct(r['tests_pass'] for r in group)} | {pct(r['false_claim'] for r in group)} "
            f"| {pct(r['placeholder'] for r in group)} "
            f"| {pct(r.get('dropped_tests') or r.get('dropped_error') for r in turn2)} "
            f"| {sum(r['prose_words'] for r in group) / len(group):.0f} | {speed} |"
        )


def main():
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        records = [json.loads(line) for line in path.open()]
        turn1_code = {}
        for record in records:
            key = (record["backend"], record["temperature"], record["run"])
            row, code = score_record(record, turn1_code.get(key) if record["turn"] == 2 else None)
            if record["turn"] == 1:
                turn1_code[key] = (code,)
            rows.append(row)
    fields = sorted({key for row in rows for key in row}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (RESULTS / "scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
