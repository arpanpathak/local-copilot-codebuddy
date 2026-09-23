"""Collects answers for the rule-following experiment.

Each run is one two-turn conversation: the task, then a follow-up that asks
for the full updated code. Answers are appended to results/<backend>.jsonl.

    python3 eval/run.py trtllm   --temps 0 0.2 0.7 --runs 2 5 5
    python3 eval/run.py llamacpp --temps 0 0.2 0.7 --runs 1 5 5

Backends:
    trtllm    Qwen2.5-Coder-7B-Instruct GPTQ-Int4, through the real
              local-copilot-codebuddy TUI (driven in a virtual terminal; the
              code is taken with Ctrl+Y, i.e. the app's own copy feature).
    llamacpp  Qwen3.5-9B UD-Q4_K_XL GGUF, through llama.cpp's llama-completion.

Requires: pyte (trtllm), xclip and an X display (trtllm), llama.cpp built in
~/.local/src/llama.cpp (llamacpp).
"""

import argparse
import fcntl
import json
import os
import pty
import re
import select
import struct
import subprocess
import termios
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RULES = (HERE / "rules.md").read_text().strip()
TURNS = [
    "Write a Rust function that parses a config file of `key = value` lines (skip blank lines and lines "
    "starting with #) into a HashMap<String, String>, with a unit test. Explain your design in detail.",
    "Now add support for `[section]` headers: keys under a section become `section.key`. "
    "Show the full updated code.",
]
# local-copilot-codebuddy's default system prompt, plus the rules as the app appends them.
SYSTEM = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Match the depth of your answer to "
    "the request: answer simple questions briefly, but when asked to explain or go into detail, write a long, "
    "thorough, well-structured answer with headings, examples and code, like a chapter of a good technical book."
    "\n\nThe user's coding rules. Follow them in every answer, for the whole conversation:\n\n" + RULES
)
# Sampling settings from Qwen's generation_config.json, used for both models.
TOP_P, TOP_K, REPETITION_PENALTY = 0.8, 20, 1.1
LLAMA_BIN = Path.home() / ".local/src/llama.cpp/build/bin/llama-completion"
GGUF = Path.home() / "models/gguf/Qwen3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf"


def clipboard():
    return subprocess.run(["xclip", "-o", "-selection", "clipboard"], capture_output=True, text=True).stdout


class TuiSession:
    """One local-copilot-codebuddy session in a virtual terminal."""

    COLS, ROWS = 200, 600

    def __init__(self, temperature):
        import pyte

        self.screen = pyte.Screen(self.COLS, self.ROWS)
        self.stream = pyte.ByteStream(self.screen)
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execvp(
                "local-copilot-codebuddy",
                ["local-copilot-codebuddy", "-t", str(temperature), "--rules", str(HERE / "rules.md")],
            )
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.ROWS, self.COLS, 0, 0))
        self.wait(lambda: any("say something" in line for line in self.screen.display), 120)

    def exited(self):
        pid, _ = os.waitpid(self.pid, os.WNOHANG)
        return pid != 0

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if ready:
                try:
                    self.stream.feed(os.read(self.fd, 1 << 16))
                except OSError:
                    return

    def status(self):
        return self.screen.display[self.ROWS - 1]

    def wait(self, predicate, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            if self.exited():
                screen = "\n".join(line.rstrip() for line in self.screen.display if line.strip())
                raise RuntimeError(f"the app exited:\n{screen}")
            self.pump(0.5)
        raise TimeoutError("the app did not respond")

    def ask(self, text):
        before = "\n".join(self.screen.display)
        os.write(self.fd, text.encode() + b"\r")
        self.wait(lambda: "generating" in self.status() or "reading" in self.status(), 120)
        self.wait(lambda: "generating" not in self.status() and "reading" not in self.status(), 900)
        self.pump(0.5)
        # The answer's prose is read from the screen; the code with Ctrl+Y (exact, unwrapped).
        subprocess.run(["bash", "-c", "printf '' | xclip -selection clipboard"])
        os.write(self.fd, b"\x19")
        self.pump(1.0)
        screen_text = "\n".join(line.rstrip() for line in self.screen.display)
        answer_screen = screen_text[screen_text.rfind("◆ assistant") :]
        match = re.search(r"(\d+) tok · ([\d.]+) tok/s", self.status())
        return {
            "code": clipboard(),
            "screen": answer_screen.split("╭ message")[0].rstrip(),
            "tokens": int(match.group(1)) if match else None,
            "tok_per_s": float(match.group(2)) if match else None,
        }

    def close(self):
        """Quits, and waits until the app has exited and given its GPU memory back."""
        os.write(self.fd, b"\x04")
        os.waitpid(self.pid, 0)
        os.close(self.fd)
        time.sleep(2)


def chatml(history):
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for role, text in history:
        prompt += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    # Qwen3.5's template in non-thinking mode (its default).
    return prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def llama_answer(history, temperature, seed):
    prompt_file = HERE / "results" / ".prompt.txt"
    prompt_file.write_text(chatml(history))
    result = subprocess.run(
        [str(LLAMA_BIN), "-m", str(GGUF), "-ngl", "99", "-fa", "on", "-c", "16384", "-n", "8192",
         "--temp", str(temperature), "--top-p", str(TOP_P), "--top-k", str(TOP_K),
         "--repeat-penalty", str(REPETITION_PENALTY), "-s", str(seed),
         "-no-cnv", "--no-display-prompt", "-f", str(prompt_file)],
        capture_output=True, text=True,
    )
    text = result.stdout.replace("[end of text]", "").strip()
    speed = re.search(r"eval time =.*?([\d.]+) tokens per second", result.stderr.split("prompt eval time")[-1])
    tokens = re.search(r"eval time =\s*[\d.]+ ms /\s*(\d+) runs", result.stderr.split("prompt eval time")[-1])
    return {
        "text": text,
        "tokens": int(tokens.group(1)) if tokens else None,
        "tok_per_s": float(speed.group(1)) if speed else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["trtllm", "llamacpp"])
    parser.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.2, 0.7])
    parser.add_argument("--runs", type=int, nargs="+", default=[2, 5, 5])
    args = parser.parse_args()

    out = HERE / "results" / f"{args.backend}.jsonl"
    out.parent.mkdir(exist_ok=True)
    for temperature, runs in zip(args.temps, args.runs):
        for run in range(runs):
            started = time.time()
            history = []
            if args.backend == "trtllm":
                session = TuiSession(temperature)
            for turn, question in enumerate(TURNS, start=1):
                history.append(("user", question))
                if args.backend == "trtllm":
                    answer = session.ask(question)
                    history.append(("assistant", answer["screen"]))
                else:
                    answer = llama_answer(history, temperature, seed=1000 + run)
                    history.append(("assistant", answer["text"]))
                record = {"backend": args.backend, "temperature": temperature, "run": run, "turn": turn, **answer}
                with out.open("a") as file:
                    file.write(json.dumps(record) + "\n")
            if args.backend == "trtllm":
                session.close()
            print(f"{args.backend} T={temperature} run {run}: {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
