from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BADGE = "https://img.shields.io/badge/{}-{}-4f5d2f?style=flat-square&labelColor=fafaf5"


def phases():
    for phase in sorted((ROOT / "phases").iterdir()):
        if not phase.is_dir():
            continue
        lines = (phase / "README.md").read_text().splitlines()
        title = lines[0].removeprefix("# ").split(": ", 1)[1]
        motto = next(l for l in lines if l.startswith("> ")).removeprefix("> ")
        lessons = []
        for lesson in sorted(p for p in phase.iterdir() if p.is_dir()):
            lessons.append((lesson, (lesson / "docs" / "en.md").read_text().splitlines()[0].removeprefix("# ")))
        yield phase, title, motto, lessons


def main():
    data = list(phases())
    total = sum(len(l) for *_, l in data)
    table = "\n".join(f"| {p.name[:2]} | [{t}](phases/{p.name}/) | {len(l)} | {m} |" for p, t, m, l in data)
    contents = []
    for phase, title, motto, lessons in data:
        rows = "\n".join(f"| {l.name[:2]} | [{name}](phases/{phase.name}/{l.name}/) |" for l, name in lessons)
        contents.append(f"""<a id="phase-{phase.name[:2]}"></a>
<details>
<summary><strong>Phase {phase.name[:2]} · {title}</strong> · {len(lessons)} lessons</summary>

> {motto}

| # | Lesson |
|---|---|
{rows}

</details>""")
    contents = "\n\n".join(contents)
    readme = f"""<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/header-dark.svg">
    <img src="assets/readme/header-light.svg" alt="Deep Learning from Scratch" width="840">
  </picture>
</p>

Build modern deep learning yourself. Start from tensors and a hand-written autodiff engine, implement every architecture from scratch, from MLPs and CNNs to transformers, diffusion models and a small GPT, then check your answer against PyTorch.

> {total} lessons. {len(data)} phases. Python and PyTorch.

<p align="center">
  <a href="LICENSE"><img src="{BADGE.format('license', 'MIT')}" alt="MIT License"></a>
  <a href="#contents"><img src="{BADGE.format('lessons', total)}" alt="{total} lessons"></a>
  <a href="#contents"><img src="{BADGE.format('phases', len(data))}" alt="{len(data)} phases"></a>
</p>

## Phases

| # | Phase | Lessons | What it is about |
|---|---|---|---|
{table}

## Every lesson has the same shape

```
NN-lesson-name/
├── code/main.py   from-scratch implementation, runnable from the repo root
├── docs/en.md     The Problem · The Concept · Build It · Use It · Exercises · Key Terms
└── quiz.json      questions to check yourself before and after
```

## Local setup

```bash
git clone https://github.com/Fer14/deep-learning-from-scratch.git
cd deep-learning-from-scratch
uv sync
uv run python phases/00-tensors-and-autodiff/01-tensors-and-broadcasting/code/main.py
```

No uv? `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` works too.

Check that every lesson runs: `uv run python scripts/check.py`

<a id="contents"></a>

## Contents

{contents}

## Contributing

New lessons follow [LESSON_TEMPLATE.md](LESSON_TEMPLATE.md). Run `uv run python scripts/check.py` before opening a pull request.

## License

[MIT](LICENSE). Part of the THOTH series, alongside [Classic ML from Scratch](https://github.com/Fer14/classic-ml-from-scratch). Structure inspired by [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).
"""
    (ROOT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
