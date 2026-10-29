# Open Detector

An AI-text detector and adversarial evaluation lab. The Streamlit app has two connected pages:

- **Detect** scores pasted text or a Word, PowerPoint or Markdown file. It gives a verdict (*Likely
  AI*, *Needs review* or *Likely human*), a calibrated AI likelihood, each engine's reading,
  sentence highlights and a score for every long section. For a worksheet it can score just the
  filled-in answer cells.
- **Adversarial Lab** generates clearly labelled AI challenge samples to probe the detector: a new
  sample from a test prompt, a section-level challenge, answers for a test worksheet, or challenges
  across a document's text blocks. Experiments can use measured style profiles as controlled
  conditions, score each sample with the detector, and save every output as AI-labelled evaluation
  data. Detector-guided runs record the target and results so they can be measured and reproduced.

The lab is for authorized detector evaluation and research. Its generated text is synthetic test
material, not human-authored work; style-conditioned generation and detector feedback are explicit
parts of the adversarial experiment. Scores are measurements against this detector build, not a
guarantee about other systems or proof of authorship.

The research loop is iterative: generate a labelled challenge, measure the detector response, add
hard examples to evaluation or training data, update the detector, then rerun the same benchmark.
The generator and detector are experimental components with measurable limits; neither is claimed
to defeat every competing system.

## Mission

Open Detector exists for scientific and engineering purposes. It is a research instrument for
studying how well AI-written text can be detected, and for building detectors that are honest about
their limits. The detector is calibrated on public benchmarks with stated false-positive rates, and
every result is reported as statistical evidence, never as proof of authorship.

The Adversarial Lab serves the same goal. Its labelled synthetic samples are test material for
measuring where the detector fails, so those failures can be reproduced, studied and fixed. Every
output is saved as AI-labelled evaluation data. The project is not a tool for passing off machine
text as human work, for evading a school's, employer's or publisher's rules, or for accusing a
person on a score alone.

Try both pages at https://open-detector.streamlit.app. In the Adversarial Lab, paste your own Gemini
key in the sidebar and add a style profile for a controlled challenge condition; both are kept only
for your browser session.

## Download the Mac app

Open Detector also comes as a Mac app that runs entirely on your computer. You don't need to install
anything else or use the Terminal. It needs a Mac with Apple silicon (M1 or newer), macOS 14 Sonoma or
later, and about 4 GB of free space.

1. Download **[Open-Detector.dmg](https://github.com/lukaberadze1/Open-Detector/releases/latest/download/Open-Detector.dmg)**.
2. Double-click the downloaded file. In the window that opens, drag **Open Detector** onto the
   **Applications** folder. If your Mac asks for an administrator password you don't have, drag it
   onto your Desktop instead.
3. Open **Open Detector** from your Applications folder or Launchpad. The first time, your Mac says
   it can't check the app for malicious software, because this free app isn't registered with Apple.
   Press **Done**, then let it open:
   - **macOS 15 or later:** open **System Settings**, go to **Privacy & Security**, scroll down to
     *"Open Detector" was blocked*, press **Open Anyway** and confirm with your password or Touch ID.
     The button shows for about an hour after you first try to open the app. macOS then asks once
     more, in a small window that can open behind others: press **Open Anyway** there too, and wait
     a minute while it checks the app.
   - **macOS 14:** Control-click (or right-click) the app, choose **Open**, then press **Open** again.

   You only do this once.
4. The app opens on its **Guide**, which explains the rest. Your first check downloads the detector,
   about 1.7 GB, once; after that it works without the internet.
5. Only for the Adversarial Lab: get a free Gemini API key at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and paste it on the Guide page.
   Checking texts needs no key.

The texts you check never leave your Mac. Your key and generated challenge samples are kept in
`~/Library/Application Support/Open Detector`, and the downloaded models in `~/.cache/huggingface`.
To update, download the newest file and replace the app; your key and samples stay. To remove it,
drag the app to the Bin and delete those two folders.

## How it scores

Three engines score overlapping windows of the whole text: GLYPH (DeBERTa-v3-base), Desklib
(DeBERTa-v3-large) and Binoculars (a Qwen2.5-0.5B pair). A logistic-regression combiner turns
their readings into one AI likelihood, with cut-offs set on held-out human writing for a 1%
(*Likely AI*) and 5% (*Needs review*) false-positive rate. It gives no verdict on texts under 150
words or on non-prose such as poems and lists, and it strips hidden characters and look-alike
letters before scoring. The *Slim* build runs Desklib alone on the CPU in about 1.5 GB.

## Run it yourself

```bash
git clone -b feat/phase1-ensemble https://github.com/lukaberadze1/Open-Detector.git && cd Open-Detector
```

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

```bash
cp .env.example .env
```

Put your Gemini key after `GEMINI_API_KEY=` in `.env` (free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)), or paste it on the app's Guide
page, which saves it there. Only the Adversarial Lab needs it.
To test style-conditioned generation, save a JSON style profile in `data/styles/`; the lab lists
every profile there and records which one was used.
Git ignores `.env` and `data/`, so your key and profile stay on your machine.

```bash
.venv/bin/streamlit run app.py
```

The app opens in your browser with **Detect** and **Adversarial Lab** in the top bar. The first run
downloads the models: 1.7 GB for Slim, about 4.4 GB for every engine. Machines with 8 GB of RAM or
less start on Slim.

On a Mac you can instead make it an app in its own window, opened from Launchpad, Spotlight or the
Dock like any other (needs the Xcode command line tools, `xcode-select --install`):

```bash
.venv/bin/python deploy/macos/build_app.py
```

That app runs your checkout. To build the self-contained download instead, `dist/Open-Detector.dmg`:

```bash
.venv/bin/python deploy/macos/build_app.py --standalone --dmg
```

To score a file from the command line:

```bash
.venv/bin/python -m opendetector essay.docx
```

## More

- [DETECTOR.md](DETECTOR.md): setup, evaluation and calibration, deployment, and the adversarial
  challenge workflow.
- [eval/reports/latest.md](eval/reports/latest.md): the current evaluation results.
- Known limits: English only; AI paraphrasing of human writing is the hardest case; a score is
  statistical evidence, not proof of who wrote a text.

This project started as a fork of followsci's Open Detector, a single BERT classifier. Its original
README is kept in [README_EN.md](README_EN.md) and [README_ZH.md](README_ZH.md).
