"""Train a small open model on the writer's own notes, to see whether text from a model that has learned the voice,
rather than been told about it, gets past the detector. What it writes is AI text and is logged as such.

    prepare   Turn the notes into pairs: a hosted model rewrites each passage in plain polished prose, twice. Resumes.
    train     On a GPU. `--method lora` learns to turn the polished version back into the writer's passage.
              `--method dpo` then prefers the writer's passage to the model's other rewrite, starting from a LoRA.
    write     Rewrite a text with the adapter, one block at a time, and log each result as labelled AI data.
    export    Merge the adapter into its base and write an Ollama Modelfile.

Only the vault's notes go in, not the finished pieces, and passages from one note stay on one side of the held-out
split. An adapter holds the writer's writing: it stays in gitignored data/adapters/ and out of public mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from opendetector.config import DATA_ROOT
from opendetector.env import load_env

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_STYLE = DATA_ROOT / "data/styles/alexandria_profile.json"
ADAPTERS = DATA_ROOT / "data/adapters"
LABELLED = DATA_ROOT / "data/essays/adapter_redteam.jsonl"

MIN_WORDS, MAX_WORDS = 40, 180  # a passage: neighbouring paragraphs of one note, about the size of a page's block
HELD_OUT = 0.1
LENGTHS = (0.6, 1.6)  # a paraphrase outside these multiples of its passage's length has lost or gained something
MAX_TOKENS = 768  # prompt and answer together; a longer pair is left out rather than cut mid-passage
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")  # the attention layers of Qwen, Llama, Mistral and Gemma
LEARNING_RATES = {"lora": 2e-4, "dpo": 5e-5}
BETA = 0.1  # DPO's: how hard it pushes away from where it started
IGNORE = -100

PARAPHRASE = ("Rewrite the passage below in clear, polished, neutral prose, as an assistant would. Keep every fact, "
              "name and number, and about the same length. Return only the rewritten passage.\n\n{text}")
REWRITE = "Rewrite this passage in your own voice. Keep every fact, name and number.\n\n{text}"
_WORD = re.compile(r"\S+")


def _words(text: str) -> int:
    return len(_WORD.findall(text))


def passages(entries: Iterable[dict], low: int = MIN_WORDS, high: int = MAX_WORDS) -> list[dict]:
    """Neighbouring paragraphs of one note joined into passages of `low` to `high` words. Superseded paragraphs and
    what is left too short are left out. A single paragraph over `high` stays whole."""
    found, note, current, size = [], None, [], 0

    def close() -> None:
        if size >= low:
            text = "\n\n".join(current)
            found.append({"id": hashlib.sha256(text.encode()).hexdigest()[:16], "note": note, "text": text})

    for entry in entries:
        if "superseded" in entry or not entry.get("text", "").strip():
            continue
        length = _words(entry["text"])
        if entry.get("note") != note or size + length > high:
            close()
            note, current, size = entry.get("note"), [], 0
        current.append(entry["text"].strip())
        size += length
    close()
    return found


def held_out(note: str | None, share: float = HELD_OUT) -> bool:
    """Whether a note's passages are kept out of training, the same way each time, as splits by group_key are."""
    return int(hashlib.sha256(str(note).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def usable_paraphrase(original: str, paraphrase: str, copied: Callable[[str, str], bool]) -> bool:
    """Whether a paraphrase is a rewrite of its passage: not the passage handed back, and about its length."""
    low, high = LENGTHS
    return (bool(paraphrase) and not copied(original, paraphrase)
            and low * _words(original) <= _words(paraphrase) <= high * _words(original))


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare(args: argparse.Namespace) -> int:
    from eval.generate_essays import PROVIDERS, APIError, clean_essay, generate_with_fallback
    from eval.redteam_batch import copied
    from eval.tune_generation import corpus_folder
    from update_style_profile import load_corpus

    load_env()
    provider = PROVIDERS[args.provider]
    api_key = os.environ.get(provider.key_env, "")
    if not api_key:
        print(f"${provider.key_env} is not set", file=sys.stderr)
        return 2
    models = [provider.model, *provider.fallbacks]
    found = passages(load_corpus(corpus_folder(args.style) / "corpus.jsonl"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {pair["id"] for pair in load_pairs(args.out)}
    print(f"{len(found)} passages, {len(done)} already paired")

    def paraphrased(text: str, temperature: float) -> str:
        reply, _, _ = generate_with_fallback(provider.base_url, api_key, models, PARAPHRASE.format(text=text),
                                             temperature, None)
        return clean_essay(reply)

    with args.out.open("a", encoding="utf-8") as out:
        for passage in found[:args.limit]:
            if passage["id"] in done:
                continue
            try:
                neutral, rejected = paraphrased(passage["text"], 0.5), paraphrased(passage["text"], 1.0)
            except APIError as exc:
                print(f"Stopped, run it again to carry on: {exc}", file=sys.stderr)
                return 1
            if not usable_paraphrase(passage["text"], neutral, copied):
                continue
            pair = {**passage, "neutral": neutral, "held_out": held_out(passage["note"])}
            if usable_paraphrase(passage["text"], rejected, copied):
                pair["rejected"] = rejected
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out.flush()
    print(f"Pairs in {args.out}")
    return 0


# Training. torch comes in only inside these functions, and the helpers take any model that returns `.logits`.

def encode(tokenizer, prompt: str, answer: str, max_tokens: int = MAX_TOKENS) -> dict | None:
    """A chat prompt and its answer as token ids, with only the answer counted in the labels. None when too long."""
    opening = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)
    start = tokenizer(opening, add_special_tokens=False)["input_ids"]
    rest = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(start) + len(rest) > max_tokens:
        return None
    return {"input_ids": start + rest, "labels": [IGNORE] * len(start) + rest}


def collate(rows: Sequence[dict], pad: int) -> dict:
    """Rows padded on the right to one tensor each."""
    import torch

    longest = max(len(row["input_ids"]) for row in rows)

    def filled(key: str, value: int):
        return torch.tensor([row[key] + [value] * (longest - len(row[key])) for row in rows])

    mask = torch.tensor([[1] * len(row["input_ids"]) + [0] * (longest - len(row["input_ids"])) for row in rows])
    return {"input_ids": filled("input_ids", pad), "labels": filled("labels", IGNORE), "attention_mask": mask}


def answer_logprobs(logits, labels):
    """Each row's total log-probability of its answer, and how many tokens that is."""
    import torch

    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    counted = labels != IGNORE
    picked = torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * counted).sum(dim=1), counted.sum(dim=1)


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta: float = BETA):
    """Direct Preference Optimization: prefer `chosen` to `rejected` by more than the starting model did."""
    import torch

    moved = (chosen - reference_chosen) - (rejected - reference_rejected)
    return -torch.nn.functional.logsigmoid(beta * moved).mean()


def batched(rows: Sequence, size: int, seed: int | None = None) -> list[list]:
    """`rows` in batches of `size`, shuffled when given a seed."""
    rows = list(rows)
    if seed is not None:
        random.Random(seed).shuffle(rows)
    return [rows[at:at + size] for at in range(0, len(rows), size)]


def _on_device(batch: dict, device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def lora_step(model, rows: Sequence[dict], pad: int, device):
    """The mean cross-entropy of a batch's answers."""
    batch = _on_device(collate(rows, pad), device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    total, tokens = answer_logprobs(logits, batch["labels"])
    return -total.sum() / tokens.sum()


def _logprobs(model, rows: Sequence[dict], pad: int, device):
    batch = _on_device(collate(rows, pad), device)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return answer_logprobs(logits, batch["labels"])[0]


def dpo_step(model, rows: Sequence[dict], pad: int, device):
    """The DPO loss of a batch of {"chosen", "rejected", "reference"} rows."""
    import torch

    chosen = _logprobs(model, [row["chosen"] for row in rows], pad, device)
    rejected = _logprobs(model, [row["rejected"] for row in rows], pad, device)
    reference = torch.tensor([row["reference"] for row in rows], dtype=chosen.dtype, device=chosen.device)
    return dpo_loss(chosen, rejected, reference[:, 0], reference[:, 1])


def with_references(model, rows: Sequence[dict], pad: int, device, size: int) -> list[dict]:
    """DPO's rows with what the starting model made of each side, worked out once here: the frozen copy of the model
    the method calls for would need twice the memory, to answer the same question every step."""
    import torch

    scored = []
    model.eval()
    with torch.no_grad():
        for batch in batched(rows, size):
            chosen = _logprobs(model, [row["chosen"] for row in batch], pad, device)
            rejected = _logprobs(model, [row["rejected"] for row in batch], pad, device)
            scored += [{**row, "reference": (float(one), float(other))}
                       for row, one, other in zip(batch, chosen, rejected, strict=True)]
    return scored


def evaluate(model, step: Callable, rows: Sequence[dict], size: int) -> float | None:
    """The mean of `step`'s loss over batches of `rows`, or None with no rows."""
    import torch

    if not rows:
        return None
    model.eval()
    with torch.no_grad():
        losses = [float(step(model, batch)) for batch in batched(rows, size)]
    return sum(losses) / len(losses)


def run_training(model, step: Callable, train: Sequence[dict], held: Sequence[dict], *, epochs: int, size: int,
                 accumulate: int, learning_rate: float, save: Callable[[], None]) -> list[dict]:
    """Train the model's trainable weights, calling `save` whenever the held-out loss is the best so far (or after
    every epoch with nothing held out). Returns each epoch's losses."""
    import torch

    weights = [weight for weight in model.parameters() if weight.requires_grad]
    optimizer = torch.optim.AdamW(weights, lr=learning_rate)
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total, batches = 0.0, batched(train, size, seed=epoch)
        for number, batch in enumerate(batches, 1):
            loss = step(model, batch)
            (loss / accumulate).backward()
            total += loss.item()
            if number % accumulate == 0 or number == len(batches):
                torch.nn.utils.clip_grad_norm_(weights, 1.0)
                optimizer.step()
                optimizer.zero_grad()
        score = evaluate(model, step, held, size)
        history.append({"epoch": epoch, "train": total / max(len(batches), 1), "held_out": score})
        print(f"epoch {epoch}: train {history[-1]['train']:.4f}, held out "
              f"{'-' if score is None else format(score, '.4f')}", flush=True)
        if score is None or score < best:
            best = float("inf") if score is None else score
            save()
    return history


def training_rows(pairs: Iterable[dict], method: str, encoded: Callable[[str, str], dict | None]) -> list[dict]:
    """Pairs as what a method trains on. DPO needs both sides, so a pair without a usable second rewrite is left out."""
    rows = []
    for pair in pairs:
        chosen = encoded(REWRITE.format(text=pair["neutral"]), pair["text"])
        if method == "lora":
            rows += [chosen] if chosen else []
        elif chosen and pair.get("rejected"):
            rejected = encoded(REWRITE.format(text=pair["neutral"]), pair["rejected"])
            rows += [{"chosen": chosen, "rejected": rejected}] if rejected else []
    return rows


def _device(requested: str, force: bool) -> str:
    import torch

    device = requested
    if requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if not device.startswith("cuda") and not force:
        raise SystemExit(f"No NVIDIA GPU here (would run on {device}). Train on a GPU machine, or pass --force to "
                         "try anyway; the 8 GB development Mac will thrash.")
    return device


def _tokenizer(base: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base(base: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    return AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device)


def train(args: argparse.Namespace) -> int:
    from peft import LoraConfig, PeftModel, get_peft_model

    if args.method == "dpo" and not args.start:
        print("--method dpo starts from a trained LoRA: pass --start data/adapters/<name>", file=sys.stderr)
        return 2
    device = _device(args.device, args.force)
    tokenizer = _tokenizer(args.base)
    pairs = load_pairs(args.pairs)
    encoded = lambda prompt, answer: encode(tokenizer, prompt, answer, args.max_tokens)  # noqa: E731
    rows = training_rows([pair for pair in pairs if not pair["held_out"]], args.method, encoded)
    held = training_rows([pair for pair in pairs if pair["held_out"]], args.method, encoded)
    if not rows:
        print(f"Nothing to train on in {args.pairs}: run prepare first", file=sys.stderr)
        return 2
    print(f"{len(rows)} training pairs, {len(held)} held out, on {device}")

    model = _load_base(args.base, device)
    if args.start:
        model = PeftModel.from_pretrained(model, args.start, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                 target_modules=list(TARGETS), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if device.startswith("cuda"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    pad = tokenizer.pad_token_id
    if args.method == "dpo":
        rows = with_references(model, rows, pad, device, args.batch_size)
        held = with_references(model, held, pad, device, args.batch_size)
        step = lambda net, batch: dpo_step(net, batch, pad, device)  # noqa: E731
    else:
        step = lambda net, batch: lora_step(net, batch, pad, device)  # noqa: E731

    args.out.mkdir(parents=True, exist_ok=True)
    history = run_training(model, step, rows, held, epochs=args.epochs, size=args.batch_size,
                           accumulate=args.accumulate, learning_rate=args.learning_rate or LEARNING_RATES[args.method],
                           save=lambda: model.save_pretrained(args.out))
    tokenizer.save_pretrained(args.out)
    (args.out / "training.json").write_text(json.dumps({
        "base": args.base, "method": args.method, "start": str(args.start) if args.start else None,
        "pairs": len(rows), "held_out": len(held), "history": history,
        "finished": datetime.now(UTC).isoformat(timespec="seconds")}, indent=2) + "\n", encoding="utf-8")
    print(f"Adapter in {args.out}")
    return 0


def _load_adapter(adapter: Path, device: str):
    from peft import PeftModel

    info = json.loads((adapter / "training.json").read_text(encoding="utf-8"))
    tokenizer = _tokenizer(str(adapter))
    return PeftModel.from_pretrained(_load_base(info["base"], device), adapter), tokenizer, info


def adapter_row(text: str, source: str, name: str, info: dict, words_kept: float | None = None) -> dict:
    """A rewrite as a labelled AI row, in the columns the evaluation reads. `source` ties it to its original."""
    key = hashlib.sha256(source.encode()).hexdigest()[:16]
    return {"id": f"adapter-{name}-{hashlib.sha256(text.encode()).hexdigest()[:16]}", "text": text, "label": 1,
            "dataset": "local_adapter", "domain": "document_block", "model": f"{info['base']}+{name}",
            "attack": f"adapter+{info['method']}", "group": "ai", "group_key": f"document-{key}",
            "words_kept": words_kept}


def write(args: argparse.Namespace) -> int:
    import torch

    from eval.redteam_batch import copied, rewrite_measures

    device = _device(args.device, args.force)
    model, tokenizer, info = _load_adapter(args.adapter, device)
    model.eval()
    blocks = [block.strip() for block in args.source.read_text(encoding="utf-8").split("\n\n") if block.strip()]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.log.open("a", encoding="utf-8") as log:
        for block in blocks:
            opening = tokenizer.apply_chat_template([{"role": "user", "content": REWRITE.format(text=block)}],
                                                    tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(opening, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                reply = model.generate(**tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                       max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(reply[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if not text or copied(block, text):  # the original handed back is the writer's text, not AI text
                continue
            kept = rewrite_measures(block, text).get("words_kept")
            log.write(json.dumps(adapter_row(text, block, args.adapter.name, info, kept), ensure_ascii=False) + "\n")
            log.flush()
            written += 1
    print(f"{written} of {len(blocks)} blocks logged as AI data in {args.log}")
    return 0


def modelfile(temperature: float = 0.8) -> str:
    """What `ollama create` reads next to a merged model's files."""
    return f"FROM .\nPARAMETER temperature {temperature:g}\n"


def export(args: argparse.Namespace) -> int:
    model, tokenizer, _ = _load_adapter(args.adapter, "cpu")
    out = args.out or args.adapter.with_name(f"{args.adapter.name}-merged")
    model.merge_and_unload().save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "Modelfile").write_text(modelfile(), encoding="utf-8")
    print(f"Merged into {out}. To serve it: cd {out} && ollama create {args.adapter.name} -f Modelfile")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("prepare", help="pair the writer's notes with polished rewrites")
    make.add_argument("--style", type=Path, default=DEFAULT_STYLE, help="the profile whose corpus to read")
    make.add_argument("--provider", choices=("gemini", "groq", "openrouter"), default="gemini",
                      help="the API to rewrite through")
    make.add_argument("--limit", type=int, help="only the first N passages")
    make.add_argument("--out", type=Path, default=ADAPTERS / "pairs.jsonl", help="resumable")
    make.set_defaults(run=prepare)

    fit = commands.add_parser("train", help="train an adapter on a GPU")
    fit.add_argument("--method", choices=("lora", "dpo"), default="lora")
    fit.add_argument("--pairs", type=Path, default=ADAPTERS / "pairs.jsonl")
    fit.add_argument("--base", default=DEFAULT_BASE)
    fit.add_argument("--start", type=Path, help="the LoRA to start from (DPO needs one)")
    fit.add_argument("--out", type=Path, required=True, help="the adapter's folder")
    fit.add_argument("--epochs", type=int, default=4)
    fit.add_argument("--batch-size", type=int, default=2)
    fit.add_argument("--accumulate", type=int, default=4, help="batches per optimizer step")
    fit.add_argument("--learning-rate", type=float, help="default 2e-4 for LoRA, 5e-5 for DPO")
    fit.add_argument("--rank", type=int, default=16, help="the LoRA's rank")
    fit.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--force", action="store_true", help="train without an NVIDIA GPU")
    fit.set_defaults(run=train)

    use = commands.add_parser("write", help="rewrite a text with an adapter, logged as labelled AI data")
    use.add_argument("adapter", type=Path)
    use.add_argument("source", type=Path, help="a text file, blocks split by blank lines")
    use.add_argument("--log", type=Path, default=LABELLED)
    use.add_argument("--temperature", type=float, default=0.8)
    use.add_argument("--top-p", type=float, default=0.95)
    use.add_argument("--max-new-tokens", type=int, default=512)
    use.add_argument("--device", default="auto")
    use.add_argument("--force", action="store_true", help="run without an NVIDIA GPU")
    use.set_defaults(run=write)

    merge = commands.add_parser("export", help="merge an adapter into its base, with an Ollama Modelfile")
    merge.add_argument("adapter", type=Path)
    merge.add_argument("--out", type=Path)
    merge.set_defaults(run=export)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
