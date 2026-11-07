# Open Detector UI inventory (for the Bistro redesign)

Every page, panel, button, dropdown, toggle, slider, text box, tab, expander, popover, table and chart in
the app, where it is drawn in the code, and what is still Glass or broken in Bistro. Written 2026-11-07
from the code, plus a run of the app in Bistro light mode at 1440 px wide (Detect with a result, Adversarial
Lab Create and Saved samples, Guide, sidebar). Dark mode and the running-batch screen were not looked at.

**Widget types used below:** *segmented* = `st.segmented_control` (a pill row of buttons), *select* =
`st.selectbox` dropdown, *multiselect* = `st.multiselect`, *toggle* = `st.toggle` switch, *slider* =
`st.slider`, *select slider* = `st.select_slider`, *text* = `st.text_input`, *text area* = `st.text_area`,
*upload* = `st.file_uploader` (or `persistent_upload`, the same with a Remove row), *expander* = a fold-out
box, *popover* = a button that opens a floating panel, *panel* = `style.group(...)` (bordered box with a
numbered small-caps title), *metric* = `st.metric` number tile.

> **Status, 2026-11-07, after the v3 build:** fixed: the base theme (config.toml is Bistro's, so sliders, links,
> tables, spinners, progress bars and pop-overs are paprika and cream; Glass repaints its own), labels above pills,
> disabled buttons, form and text-only buttons, one mono heading style, the bill and hygiene at one height, the Lab's
> result actions in one row, Model & connection under the order button, `/detect`, the Guide's Look text, the violet
> badge and dots (Bistro's priorities are now ingredients). Added from the v3 boards: Cloche in the corner with
> reactions and lines, the bell, the kitchen timer, the needle, the centred-then-slide layout, the Textured/Standard
> finish. Not done: the v2 boards' Settings page and the Lab's three mode cards. The list below is as it was.

---

## Part 1 · What's wrong in Bistro right now

### A. The root cause: Streamlit's own theme is still Glass

[.streamlit/config.toml](../.streamlit/config.toml) sets the base theme every native Streamlit widget uses,
and it is still Glass's: violet `primaryColor = "#5B57F5"` (dark: `#8F8CFF`), lavender-grey backgrounds
(`#EEF0F8`, `#F7F8FC`), lavender borders (`#DADDEA`), Geist fonts. Its own comment still says "Glass by
default, or Instrument from Settings". `ui/style.py` only paints over *some* widgets for Bistro, so anything
it doesn't name shows the violet/lavender underneath. Measured in the running app, these were still violet
or lavender in Bistro:

| Where it shows | What's Glass about it |
|---|---|
| Every **slider** (Approximate length, Only revise passages above, Revise passages above, Words per block, Creativity, Blocks per request, Requests at once, Top-p, Top-k) | Filled track and the value label over the thumb ("350 words", "0%") are violet. Only the thumb is restyled. |
| **Select slider** "Model thinking" | Same violet track and label. |
| Every **link** (Guide's aistudio link, captions with URLs like `http://localhost:11434`) | Violet `linkColor`. |
| **Tabs** (Detect result tabs, Lab Create / Saved samples) | Active tab's own colour and focus outline are violet (the label text is repainted, the focus ring isn't). |
| **Segmented controls** | The selected button's base colour, border and keyboard focus outline are violet under the espresso fill. |
| **Tables** (`st.dataframe`: Sections, Sentences, Details features, File check properties, Saved samples tables, Style at each stage) | Header row and grid lines are the cool lavender-grey of the Glass palette, not cream. |
| **Form submit buttons**: Guide "Save key" / "Replace key", Lab "Check and save key", "Check and add key" | Fully Glass: a violet pill. `BISTRO_CSS` styles `stBaseButton-primary` but not `stBaseButton-primaryFormSubmit` / `secondaryFormSubmit`. |
| **Tertiary buttons** (Clear selection, Remove, Forget it on this device, Discard and start over, the priorities' − and +) | No Bistro style at all. |
| **Popovers** ("N Gemini keys", "N models on offer") | Floating panel uses Streamlit's default surface and violet focus. |
| **Progress bars** (Scoring sections…, key allowance bars, Glass batch progress) | Violet fill. |
| **Multiselect** chips ("Other sections to give as context") | Violet/default tag chips. |
| **Spinners** and **st.status** boxes ("Tasting in 350-word bites…", "Scoring the passages…") | Violet spinner ring. |
| **Badges** in the Lab's Latest result | Attack badge is hard-coded `color="violet"` ([redteam.py:2863](redteam.py:2863)); model badge is grey. |
| **Priority dots** (●●●○○○ beside each priority) | Hard-coded `:violet[...]` ([redteam.py:1467](redteam.py:1467)). |
| **Expander header when open/hovered** | Lavender tint from the base theme. |
| **Mono fonts**: page section headings drawn with `st.markdown("**…**")` or `st.subheader` ("Latest result", "By attack", "Latest challenge samples", "What moved the score", the "Result" title) | Plain Geist bold, while panel titles are JetBrains Mono. Guide's `st.subheader`s go the other way: huge mono. |

### B. Layout and behaviour errors seen in the app

1. **Labels trapped inside the pill track.** A segmented control with a visible label (Lab "Engine",
   "Register", "How closely to follow the original", "Rewrite" in the revise panel) draws its label *inside*
   the beige track pill, so "Engine ⓘ" sits in the pill above the buttons. `[data-testid="stButtonGroup"]`
   now wraps the label too; the background should go on the inner button row instead.
2. **Disabled primary button looks active.** "Revise selected (0)" on Detect is disabled but still full
   paprika with the pressed shadow. Bistro has no `:disabled` style for any button.
3. **Lab result actions don't line up.** "Text file" (left column) and "Another adaptive round" /
   "Analyze in Detect" (right column) sit on different rows, the two right-hand buttons stacked.
4. **Bill and Kitchen hygiene panels** on Detect are different heights side by side.
5. **Guide's two top panels** ("1 · Check a text", "2 · Read the result") are different heights.
6. **Model & connection** is a long expander (provider, URL, model, fallbacks, Ollama, keys, ~10 fields)
   tucked *under* the Detector panel and above the Generate button, so opening it pushes Generate far down.
7. **The page is in Glass because your saved look is Glass.** `data/ui_prefs.json` holds
   `{"look": "glass"}`, which wins over the Bistro default. Pick Bistro under **Look** at the foot of the
   sidebar. (I switched it to check Bistro and put it back to Glass afterwards.)
8. **`/detect` gives "Page not found".** Detect is the default page, so its address is `/`; `url_path="detect"`
   is ignored. Any link or bookmark to `/detect` hits the not-found dialog.
9. **Stale text:** Guide "Good to know" says *"Look: Glass or Instrument, under Settings at the bottom of the
   menu"* ([guide.py:156](guide.py:156)). It's now Bistro or Glass, under **Look**. `config.toml`'s comment
   and `.od-hero-title`'s gradient comment also still describe Glass/Instrument.
10. **Guide uses a different header** (`style.hero`, big title, chips) than Detect and the Lab
    (`style.page_header`, eyebrow + title + status pill), and its eyebrow uses the Glass pulse dot.
11. **Primary button text colour:** `BISTRO_CSS` forces `#FFFCF6` on every child of a primary button, which
    in dark mode puts cream on `#EA6A2E`. Worth checking contrast in dark mode.
12. **Bistro-only components vs Glass-only components** (both kept, so a redesign has two to maintain):

    | Thing | Bistro draws | Glass draws | Switch in code |
    |---|---|---|---|
    | Verdict | Cloche the critic + cut-off band | Ring dial | [components.py:460](components.py:460) |
    | Engine readings | "The bill" + "Kitchen hygiene" | Engine meters panel | [detect.py:234](detect.py:234), [components.py:520](components.py:520) |
    | Running batch | Order ticket of stations | Metric tiles + progress bars | [redteam.py:1967](redteam.py:1967) |
    | Chart font | JetBrains Mono | Geist | [components.py:731](components.py:731) |
    | Loading/empty lines | Kitchen phrases (`style.flavour`) | Plain | style.py:112 |

---

## Part 2 · Everything on screen, page by page

Line numbers point at the code that draws each item. **Order = top to bottom as it appears.**

### 0 · App shell (every page) — [app.py](../app.py), [ui/style.py](style.py), [ui/nav.py](nav.py)

| # | Element | Type | Label / content | Code |
|---|---|---|---|---|
| 0.1 | Browser tab | page config | "Open Detector", mark icon | app.py:23 |
| 0.2 | Sidebar logo | `st.logo` | `logo-bistro[-dark].svg` / `logo[-dark].svg` (Glass) | app.py:24, style.py:137 |
| 0.3 | Sidebar collapse `>>` | Streamlit built-in | | |
| 0.4 | Nav link | page link | **Detect** (kitchen line "The critic's table"), url `/` | nav.py:13, style.py:344 |
| 0.5 | Nav link | page link | **Adversarial Lab** ("The kitchen"), url `/red-team`; hidden with `OPENDETECTOR_REDTEAM=off` | nav.py:28 |
| 0.6 | Nav link | page link | **Guide** ("The menu"), url `/guide` | nav.py:34 |
| 0.7 | Look box | segmented (container `od-look`) | "LOOK": **Bistro / Glass** | style.py:126 |
| 0.8 | Top-right | Streamlit built-in | **Deploy**, **⋮** menu (Settings → light/dark, Rerun, etc.) | |
| 0.9 | Page header | HTML `.od-page` | kitchen eyebrow, title, subtitle, status pill (green dot) | style.py:144 |
| 0.10 | Panel | HTML `.od-group` in a bordered container | "01 · TITLE · kitchen name" | style.py:159 |
| 0.11 | Empty state | HTML `.od-empty` | dashed box, "Nothing on the plate yet" | |

Not shown: Guide is opened first once per computer (app.py:38), and a public deploy hides the Guide's
key-saving.

### 1 · Detect page — [ui/detect.py](detect.py)

Header: eyebrow "THE CRITIC'S TABLE", title **Check a text**, subtitle "Paste text or upload a document, then
press Analyze.", pill "Slim build · 1 of 1 engines on" (detect.py:540).
Two columns: left (settings) 1, right (result) 1.45 (detect.py:544).

**Left column**

| # | Element | Type | Label / options | Code |
|---|---|---|---|---|
| 1.1 | Panel | panel | **01 · Source** | 546 |
| 1.2 | Input switch | segmented | **Paste text / Upload a file** | 90 |
| 1.3 | Text box | text area, 300 px | placeholder "Paste an essay, a report or a draft…" | 93 |
| 1.4 | Upload (when Upload a file) | upload + Remove row | "Markdown, Word or PowerPoint file" (.md .docx .pptx) | 101, components.py:848 |
| 1.5 | Remove uploaded file | tertiary button | **Remove** | components.py:862 |
| 1.6 | Reference material | expander | "Reference material" → multi-file upload, never scored | 74 |
| 1.7 | Answers only | toggle (worksheets only) | "Score only the N filled-in answer cells" | 113 |
| 1.8 | Extracted text | expander | "Extracted text · N sections" | 122 |
| 1.9 | Word count | caption | "178 words" | 550 |
| 1.10 | Panel | panel | **02 · Engines** | 52 |
| 1.11 | Build | segmented | **Full build / Slim build** (caption under it); hidden when the profile is pinned | models.py:75 |
| 1.12 | Engine switches (Full only) | toggles in a row | **GLYPH**, **Desklib**, **Binoculars** | 60 |
| 1.13 | Sentence highlights | toggle | "Sentence highlights" | 66 |
| 1.14 | **Analyze** | primary button, full width | | 553 |
| 1.15 | Integrity note | caption | "A detector score is statistical evidence…" | 554 |

**Right column (result)**

| # | Element | Type | Label / content | Code |
|---|---|---|---|---|
| 1.16 | Empty state | `.od-empty` | "Nothing on the plate yet" | 569 |
| 1.17 | Loading | spinners | "Loading Desklib…", "Tasting in 350-word bites…", progress "Scoring sections…" | 133, 141, 167 |
| 1.18 | Revise in the Lab (only when picking is off) | secondary button | **Revise flagged passages in the Lab** | 450 |
| 1.19 | Result title | HTML `.od-section-title` | "Result · Pasted text · Slim build" | 454 |
| 1.20 | Stale note | caption | "The input has changed since this analysis…" | 457 |
| 1.21 | Warnings | error / warning / info | hidden characters found; engines off; too short; not prose | 461–479 |
| 1.22 | Verdict card | HTML | Bistro: Cloche + "THE CRITIC SAYS · Likely AI · 100% AI likelihood" + band; Glass: dial | components.py:445 |
| 1.23 | The bill (Bistro) | HTML, left of two columns | engine %, words, sentences, words flagged, TOTAL | 240, components.py:513 |
| 1.24 | Kitchen hygiene (Bistro) | HTML, right column | hidden characters result | 246 |
| 1.25 | Engine meters (Glass) | HTML | same data as bill | 235 |
| 1.26 | Result tabs | tabs | **Highlights · Sentences · Sections · File check · Details** (only those that apply) | 486 |

*Highlights tab*

| # | Element | Type | Label | Code |
|---|---|---|---|---|
| 1.27 | Highlight filter | segmented | **Every sentence / Flagged only** | 490 |
| 1.28 | Revise bar | button row | **Revise selected (N)** primary · **Revise all Likely AI (N)** · **Revise all flagged (N)** (only when more than Likely AI) · **Clear selection** tertiary | 389–403 |
| 1.29 | Revise settings | expander | "Revise settings · style profile and model": the Lab's Writing settings + Hold names and numbers toggle + The source is AI text toggle (see 2.x) | 404, redteam.py:1068 |
| 1.30 | Revise progress | status box | "Round 1 of up to N: …" | 414 |
| 1.31 | Revise result | success + **Undo** button | "Revised N passages: …" | 369–376 |
| 1.32 | Legend + map | HTML | Likely AI / Needs review / Likely human shares, start→end map bar | components.py:572 |
| 1.33 | Highlighted text | custom component (clickable sentences) | | components.py:709 |
| 1.34 | Note | caption | "Stronger shades sit deeper in their band…" | 503 |

*Sentences tab*: caption; bar chart (Altair) of every sentence; table #, AI likelihood (progress column),
Band, Sentence (510–523).
*Sections tab*: caption; bar chart; table Section / Words / Verdict / AI likelihood; **select**
"Read a section" (placeholder "Choose a section to read") → bordered box with verdict badge + text (252–272).
*File check tab* (uploads only): caption; red error / blue info rows per finding, or green "Nothing
unusual"; Property/Value table (296–309).
*Details tab*: "What moved the score" chart (2+ engines); Build/Engines/Calibration/Held-out/Cut-offs lines;
Feature/Value/In the score table; two captions (312–344).

### 2 · Adversarial Lab page — [ui/redteam.py](redteam.py)

Header: eyebrow "THE KITCHEN", title **Adversarial Lab**, subtitle "Make labelled AI test text…", pill
"Scored with the Slim build" (3099).
While a batch runs, the page shows **only** the batch screen (2.40). Otherwise two tabs (3116):
**Create · Saved samples**. Create has two columns, settings 1 : result 1.2 (3118).

**Create → left column, top to bottom** (`_setup`, 1284)

| # | Element | Type | Label / options | Code |
|---|---|---|---|---|
| 2.1 | Panel | panel | **01 · Mode** | 1289 |
| 2.2 | Mode | segmented, full width | **Generate / Regenerate** + caption | 724 |
| 2.3 | Engine (not public) | segmented with label | "Engine": **API / Math / Combo** | 1292 |
| 2.4 | Panel | panel | **02 · Input** · the ingredients | 1297 |
| 2.5 | Instructions | text area 120 px | "Instructions" / "Instructions (optional)" | 732 |
| 2.6 | References (Generate only) | text area 100 px | "References for citations (optional)", JSON | 738 |
| 2.7 | Source file | upload + Remove | "Source material (optional)" / "Document to rewrite" | 759 |
| 2.8 | Pasted source (when no file) | text area 110 px | "Or paste source text (optional)" / "Or paste the text to rewrite" | 767 |
| 2.9 | Extracted text | expander | "Extracted text · N sections" | 790 |
| 2.10 | Panel | panel | **03 · Priorities** · the head chef | 1305 |
| 2.11 | Own settings | toggle | "Write my own settings" | 1447 |
| 2.12 | Four priority rows | markdown + dots + two tertiary buttons each | 🎯 Detector challenge (lower score) · Style-profile match · Speed · Saves quota, each **−** / **+** | 1462–1474 |
| 2.13 | Points left | caption | "0 of 12 points left…" | 1475 |
| 2.14 | Diamond chart + readout | HTML image + text, two columns | DETECTOR / STYLE / SPEED / QUOTA; picked model, time, requests | 1489 |
| 2.15 | Hand settings (own settings on, or nothing measured) | slider, select slider, 2 sliders | **Creativity** 0–1.5 · **Model thinking** (levels) · **Blocks per request** 1–10 · **Requests at once** 1–5 (batches only) | 1543–1566 |
| 2.16 | Panel (Regenerate only) | panel | **04 · Rewrite** · what goes in the pan | 1308 |
| 2.17 | What to rewrite | select | **Whole document / One section / Answer boxes only** + caption, "isn't available" captions | 811 |
| 2.18 | Section (One section) | select | "Section to rewrite" | 822 |
| 2.19 | Context (One section) | multiselect | "Other sections to give as context" | 825 |
| 2.20 | Export (One section) | toggle | "Also export a copy of the document with this section replaced" | 832 |
| 2.21 | Answer boxes (Answer boxes only) | expander | "Detected answer boxes · N" | 837 |
| 2.22 | Also answers (Whole document) | toggle | "Also fill the N answer boxes" | 846 |
| 2.23 | Blocks list (Whole document) | expander | "Blocks to rewrite · N" | 854 |
| 2.24 | Closeness (Whole document) | segmented with label | "How closely to follow the original": Light edit / Same structure / Writer's sentences / Free rewrite | 858 |
| 2.25 | Own writing | toggle | "This document is my own writing" | 868 |
| 2.26 | Panel | panel | **04/05 · Writing** · the recipe card | 1310 |
| 2.27 | Length (sample) | slider | "Approximate length" 150–1500 words | 890 |
| 2.28 | Match lengths (batch) | toggle | "Match the length of the existing text" | 883 |
| 2.29 | Words per block (batch) | slider | "Words per block" 20–500 | 885 |
| 2.30 | Use profile | toggle | "Write in a style profile" | 893 |
| 2.31 | Profile | select | "Style profile" (your profiles, "Human (built in)") | 905 |
| 2.32 | Register | segmented with label | "Register": **Finished work / Quick notes** | 1353 |
| 2.33 | Examples | toggle | "Show the model the writer's own passages" | 1428 |
| 2.34 | Restyle | toggle | "Rewrite the draft in the writer's sentences" | 1407 |
| 2.35 | Add profile | expander → upload | "Add your own style profile" → "Style profile (.json)" | 919 |
| 2.36 | Panel | panel | **05/06 · Detector** · the critic | 1312 |
| 2.37 | Score | toggle | "Score with the detector" | 932 |
| 2.38 | Adaptive (not public) | toggle | "Adaptive challenge mode · detector score + style profile" | 936 |
| 2.39 | Revise | toggle | "Revise what the detector flags" | 948 |
| 2.39a | Revise threshold (sample only) | slider | "Only revise passages above" 0–95% | 959 |
| 2.39b | Hold facts (own settings) | toggle | "Hold names and numbers exactly" | 966 |
| 2.39c | More attacks (own settings) | expander | "More attacks": select "Round trip through a language", select "Round trip on this computer", toggle "Have the model rewrite it in that language first", toggle "Move clauses around", toggle "Swap stock phrases and linking words", toggle "Free translation round trip", sliders "Top-p", "Top-k" | 1231–1275 |
| 2.39d | Math/Combo | toggles | "Use language models too", "The source is AI text" (Math) | 1318, 1325 |
| 2.39e | Model & connection | expander | "Model & connection · the cooks" (see 2.41) | 1330 |
| 2.39f | **Generate** | primary button, full width | changes to **Regenerate N blocks** / **Regenerate section** / **Resume · N of M blocks saved** | 3139 |
| 2.39g | Saved batch notice | info + tertiary button | **Discard and start over** | 2112 |
| 2.39h | Revise a text | expander | "Revise a text above a percentage": text area "Text", slider "Revise passages above", segmented "Rewrite" (With the model / On this computer), toggle "The source is AI text", button **Revise**, then metrics Before / After / Requests and "Revised text" box | 1127–1160 |
| 2.39i | Evaluation note | caption | "Evaluation use only…" | 3155 |

**2.41 · Model & connection (inside the expander)**, local run (`_connection`, 535):
status captions (model, keys, Ollama ready/not) at the top; **popover** "N Gemini keys" (key rows with
**Remove** tertiary, allowance progress bars, form text "Add a key from another account" + **Check and add
key**); **popover** "N models on offer" (Groq/OpenRouter); page link "Add your Gemini API key"; key form
"<Provider> API key" + **Check and save key** (Groq/OpenRouter without a key); select **Provider**
(Gemini / Groq / OpenRouter / Other); text **API URL**; text **Model**; text area **Fallback models, in
order**; text **Local fallback model (Ollama)**; text **Local fallback URL**; text **API key environment
variable**.
Public mode (`_public_connection`, 679): password text "Your Gemini API key", toggle "Remember it on this
device", tertiary **Forget it on this device**, text **Model**, text area **Fallback models, in order**.

**Create → right column (result)**

| # | Element | Type | Label | Code |
|---|---|---|---|---|
| 2.42 | Empty state | `.od-empty` | "Nothing on the plate yet" | 3174 |
| 2.43 | Result box | bordered container | **Latest result** + grey model badge + violet attack badge | 2855 |
| 2.44 | Notes | captions | "The detector abstained…", adaptive notes | 2864 |
| 2.45 | Scores | metrics | AI likelihood, Style match, Structure match (before/after when there's a baseline) | 2875–2911 |
| 2.46 | Style stages | expander → table | "Style at each stage" | 2896 |
| 2.47 | Batch blocks | table | per-block scores | 2918 |
| 2.48 | Sample text | HTML `.od-text` | | 2929 |
| 2.49 | Downloads | download buttons | **Text file**, **BibTeX**, **References JSON**, **Document copy (.docx)** | 2933–2946 |
| 2.50 | Next steps | buttons | **Another adaptive round** (secondary), **Analyze in Detect** (primary) | 2949–2961 |
| 2.51 | Full breakdown | expander | "Full breakdown" → the whole Detect result (1.19–1.34) inside | 2837 |
| 2.52 | Math result (Math engine) | metrics + text area + download | "AI likelihood before/after…", "Edited text", **Download** | 1210–1226 |

**2.40 · Running batch screen** (replaces Create while a batch runs): note text; Bistro **order ticket**
(Cloche "tasting", stations Unpack → Hold names and numbers → Head chef plans → Recipe card → Cooking →
Inspection → Critic's first taste → Extra techniques → Refining → Plating, progress bar) or Glass metrics
(Revising round, Passages tested, AI likelihood now, Time left, Done around, Blocks written) + progress bars;
log captions; **Stop** button → disabled "Stopping after the blocks in progress…" (1952–2067).

**Saved samples tab** (`_history`, 2982): metrics **Samples · Your originals · Scored · Passed as human ·
Flagged as AI**; heading "By attack" + table (Attack, Scored, Passed as human bar, Mean AI likelihood bar);
heading "Latest challenge samples" + table (Saved, Model, Attack, Kind, Words, AI likelihood, Verdict, Text);
caption with the file path.

### 3 · Guide page — [ui/guide.py](guide.py)

| # | Element | Type | Label | Code |
|---|---|---|---|---|
| 3.1 | Hero | HTML `.od-hero` | eyebrow "THE MENU" (pulse dot), **How Open Detector works**, lede, chips "Free and open source", "Works offline after the first check", "Word, PowerPoint, Markdown" | 119 |
| 3.2 | Panel (left) | keyed container `od-panel-guide-check` | subheader "1 · Check a text", steps | 127 |
| 3.3 | Panel (right) | `od-panel-guide-read` | "2 · Read the result" | 140 |
| 3.4 | Key panel | `od-panel-guide-key` | "Your Gemini key, for the Adversarial Lab", steps with link | 51 |
| 3.5 | Saved key | success | "A key is saved, ending in …" | 64 |
| 3.6 | Key form | password text + form submit | placeholder "Paste your key here"; **Save key / Replace key** (violet) | 67–71 |
| 3.7 | Checking | spinner | "Checking the key with Google…" | 78 |
| 3.8 | Remove | secondary button | **Remove the key** | 88 |
| 3.9 | Public variant | page link | **Open the Adversarial Lab** | 114 |
| 3.10 | Panel | `od-panel-guide-more` | "Good to know" (stale Look text) | 153 |
| 3.11 | **Start checking texts** | primary button | goes to Detect | 167 |

---

## Part 3 · Where the styling lives

| File | What to change there |
|---|---|
| [.streamlit/config.toml](../.streamlit/config.toml) | Base colours/fonts every native widget uses (the violet). Note it applies to *both* looks, so Bistro colours here would make Glass's native widgets paprika unless Glass CSS repaints them. |
| [ui/style.py](style.py) `TOKENS` | Each look's colours per light/dark (`bg`, `panel`, `edge`, `ink`, `accent`, `side`, `dash`, `press`, …). |
| [ui/style.py](style.py) `BASE_CSS` | Shared CSS, mostly Glass (blur, rounded pills, glow). Bistro builds on top of it. |
| [ui/style.py](style.py) `BISTRO_CSS` | Bistro overrides. Widgets it currently covers: sidebar + nav links, Look box, text areas/inputs/select box, file dropzone, primary + secondary buttons, segmented controls, toggles/checkboxes, slider thumb only, tabs, expanders, data frames (border only), page header, panels, empty state, hero, verdict card. Not covered: form-submit, tertiary and disabled buttons, slider track/value, select slider, progress, spinner, status, popover, multiselect, badge, links, metric tiles, dataframe body, alerts, `st.subheader`/markdown headings. |
| [ui/style.py](style.py) `PAGE_FLAVOUR`, `group(kitchen=)`, `flavour()` | Kitchen names under nav links, over titles, on panels, in loading/empty lines. |
| [ui/components.py](components.py) | Verdict card, Cloche (`CLOCHE_TONES`, `MOODS`), dial, bill/engine panel, order ticket, highlight HTML + its CSS (`CSS`), verdict tones (`TONES`, `BISTRO_TONES`; colour-blind checked, keep the underline styles), Altair charts (`_dress`), `persistent_upload`. |
| [ui/redteam.py](redteam.py) | Hard-coded `:violet[...]` dots (1467), `color="violet"` badge (2863), `:orange[...]` captions (591, 704), panel order in `_setup` (1284), result action row (2931). |
| [ui/guide.py](guide.py) | Stale "Glass or Instrument" text (156), hero vs page header. |
| `ui/assets/*.svg` | Logos and marks per look. |

Streamlit `data-testid` names to target for the uncovered widgets: `stBaseButton-primaryFormSubmit`,
`stBaseButton-secondaryFormSubmit`, `stBaseButton-tertiary`, `stSlider` (track `[data-baseweb="slider"]`),
`stSliderThumbValue`, `stProgress`, `stSpinner`, `stStatusWidget`/`stExpander` for `st.status`,
`stPopover`/`stPopoverButton`, `stMultiSelect` (`[data-baseweb="tag"]`), `stBadge`/`stMarkdownBadge`,
`stMetric`, `stDataFrame` (glide grid takes its colours from the theme, so config.toml is the real fix),
`stAlertContainer`, `stPageLink`. Check each against the installed Streamlit version before relying on it.
