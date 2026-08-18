# Climate justice classifiers — exploratory analysis

> **Written by Claude (Anthropic), working with Anne Sietsma, August 2026.**
> Every number here was produced by the scripts in this directory and can be
> regenerated from them. The analysis, the framing and the figure captions are
> machine-written and have not been independently reviewed — treat the findings
> as a starting point for discussion rather than as settled results, and check
> anything you intend to publish against the source data. Points where the
> evidence does not settle a question are flagged in the text rather than
> smoothed over.

Three BERT classifiers, all `primary` profile, all carrying
`dont_run_on: ["sabin"]`:

| Concept | Label | Body-text passages |
|---|---|---:|
| `Q32` | climate justice | 420,780 |
| `Q911` | distributive justice | 748,528 |
| `Q912` | procedural justice | 230,667 |

898,556 body-text passages carry at least one justice label. Litigation is
excluded throughout: all three specs carry `dont_run_on: ["sabin"]` and every
published Litigation document is a Sabin record (17,347 of 17,347), so there is
no inference there to report.

Eleven figures in `figures/`, all numbers reproducible from `pull_data.py` →
`text_stats.py` / `specificity.py` → `viz.py`.

---

## 1. The three classifiers have genuinely distinct vocabularies

Log-odds ratio with informative Dirichlet prior, each class against the other
two, on *exclusive* passages (fig 1, 12 single words + 6 two-word terms per class):

- **Climate justice (Q32)** — `women` (z=102), `gender` (86), `human` (80),
  `rights` (76), `social`, `men`, `female`, `vulnerable`. The recognition and
  rights strand: who is affected and who gets named. It also carries a strong
  fund-reporting signal (`reporting period`, `indicator`, `evaluation`), which
  follows from Multilateral Climate Fund projects having the highest hit rate
  of any corpus.
- **Distributive justice (Q911)** — `energy` (127), `emissions` (94),
  `renewable` (65), `production`, `efficiency`, `carbon`, `sector`, `economy`,
  `waste`. See §2.
- **Procedural justice (Q912)** — `stakeholders` (109), `community` (92),
  `consultation` (85), `representatives` (73), `participation`, `meetings`,
  `committee`, `council`, `engagement`. The cleanest separation of the three.

**One- and two-word terms are ranked in separate vocabularies.** A bigram is
necessarily rarer than either of its parts, so in a pooled ranking the Dirichlet
prior shrinks it harder and almost none survive into the top slots — `civil
society` and `local communities` were invisible behind `community` and `local`.
Ranked apart, the two-word terms are often the more legible half:

| | Top two-word terms |
|---|---|
| Q32 | reporting period (65), environmental social (48), annual performance (31), performance report (31), green climate (31), climate fund (30) |
| Q911 | energy efficiency (52), renewable energy (51), climate change (48), ghg emissions (43), greenhouse gas (42), gas emissions (38) |
| Q912 | civil society (48), local communities (36), working group (36), key stakeholders (34), local authorities (32), local government (31) |

Q912's bigrams are the sharpest statement of what that classifier is: the
institutional furniture of consultation. Q32's are dominated by fund-reporting
boilerplate, which is worth watching — `reporting period`, `annual performance`,
`performance report` are artefacts of where the classifier fires (MCF projects),
not of justice language as such.

TF-IDF over merged class documents agrees on direction but discriminates less
well — on merged documents it rewards globally rare terms, so Q911 and Q912
both come back dominated by generic frequent words. Quote the log-odds.

## 2. What distributive justice is actually picking up

Q911's distinctive vocabulary is the vocabulary of mitigation policy — energy
systems, emissions, renewables, efficiency — not the vocabulary of allocation
(revenue, compensation, burden-sharing, affordability). It fires on passages
like Albania's energy strategy or Liechtenstein's agriculture law.

**This is a finding about the corpus, not a defect in the classifier.** The
whole point of training a BERT model rather than a keyword list was to catch
distributive claims that never say "equity". What the vocabulary shows is
*where* those claims sit in this corpus: overwhelmingly in decisions about who
builds and who pays for the energy transition, rather than in explicit
discussion of payment, risk-sharing or the distribution of adaptation burdens.

A crude lexicon probe (`specificity.py`) puts a number on how little explicit
equity language is involved:

| Class | Explicit justice/equity | Distributive markers | Procedural markers | Recognition markers | Any of four |
|---|---:|---:|---:|---:|---:|
| Climate justice (Q32 only) | 12.0% | 9.3% | 24.1% | 45.0% | 70.2% |
| Distributive justice (Q911 only) | 1.3% | 13.4% | 9.5% | 12.6% | 32.5% |
| Procedural justice (Q912 only) | 2.2% | 4.3% | 56.0% | 8.5% | 63.1% |
| All three | 16.5% | 13.5% | 61.3% | 58.9% | 88.7% |
| *Corpus background (no justice label)* | *3.0%* | *6.0%* | *6.3%* | *4.5%* | *17.9%* |

**The lexicons, in full**, so the numbers can be judged. Substring matching,
lowercased, one hit anywhere in the passage counts:

- *Explicit justice / equity* — justice, equity, equitable, equitably, fair,
  fairness, unfair, inequality, inequalities, inequitable, just transition,
  injustice
- *Distributive markers* — distribution, distributive, redistribut-, allocate-,
  revenue, subsidy, subsidies, compensat-, burden, benefit-sharing, benefit
  sharing, affordab-, low-income, low income, poverty, the poor, disadvantaged,
  cost of living, energy poverty
- *Procedural markers* — consultation, consult, participants-, stakeholder,
  engagement, consent, transparen-, accountab-, grievance, representation,
  public hearing, deliberat-, co-design, inclusive decision
- *Recognition markers* — indigenous, traditional knowledge, customary,
  cultural, women, gender, youth, marginali-, vulnerable, minorit-, disabilit-,
  ethnic, tribal, first nations

These were written by hand before looking at the results, and they are a
yardstick rather than ground truth — a passage can be squarely about
distributive justice without using any of these words, and the classifiers were
built on that premise. Read the table as "how much of each classifier's output
would a keyword list have found", not as a precision score. The honest summary:
a keyword list would have recovered most of what Q912 fires on, about
two-thirds of Q32, and under a third of Q911. Whether Q911's remaining
two-thirds is signal a keyword list would miss or noise a keyword list would
correctly reject is the question a labelled precision audit would answer;
these word counts cannot settle it.

## 3. Overlap: distributive justice does most of the work, alone

Of the 896,744 justice-labelled passages (fig 2, UpSet — the seven combinations
are disjoint and sum to 100%):

| Combination | Passages | Share |
|---|---:|---:|
| Distributive only | 403,043 | 44.9% |
| Climate + distributive | 185,972 | 20.7% |
| All three | 122,844 | 13.7% |
| Climate only | 77,807 | 8.7% |
| Procedural only | 38,604 | 4.3% |
| Distributive + procedural | 35,119 | 3.9% |
| Climate + procedural | 33,355 | 3.7% |

54% of all Q911 hits fire with no other justice label attached; procedural
justice is the most dependent, with only 17% firing alone.

## 4. Corpus and geography

**By corpus (fig 3).** Cells are each corpus's own passage share, so corpus size
is divided out; shading is scaled within each column because the three
classifiers have very different base rates (12.0% / 21.3% / 6.6% overall).
Multilateral Climate Fund projects lead on all three (28.1% / 32.4% / 16.6%).
Policy is close behind on distributive (30.0%) but middling on procedural
(6.5%). Law runs 6.0% / 10.6% / 2.8% and Corporate Disclosure is the floor at
5.8% / 5.1% / 2.2%. Only Litigation is excluded, because the classifiers were
never run on it.

**By region (fig 4).** Sub-Saharan Africa leads on all three, Europe & Central
Asia and North America trail. Two confounds tested:

- *Fund documents.* Removing Multilateral Climate Fund projects — the densest
  corpus, concentrated in the leading regions — barely moves distributive
  justice (SSA:ECA ratio 1.84× → 1.75×) but cuts a lot off climate justice
  (SSA 21.0% → 14.3%) and procedural justice (11.8% → 7.9%). So fund reporting
  was inflating those two strands specifically, not the gradient as a whole.
- *Corpus mix.* The ranking survives within a single corpus: SSA leads ECA on
  distributive justice inside UN submissions (25.2% vs 12.0%) and inside Policy
  (38.9% vs 28.7%), though the gap narrows sharply in the latter.

## 5. Deep dives

**Seven documents (figs 5a and 5b).** Three 2026 NBSAPs, plus both countries' current NDC
and current long-term strategy. Tick width is one passage's share of the
document, so equal ink means an equal share — a 54-passage NDC and a
2,697-passage strategy no longer look equally dense at very different rates.

| Document | Passages | Any justice | Q32 | Q911 | Q912 |
|---|---:|---:|---:|---:|---:|
| Türkiye NDC 3.0 (Nov 2025) | 54 | **51.9%** | 37.0% | 46.3% | 16.7% |
| Australia 2035 NDC (Sep 2025) | 155 | **45.2%** | 12.9% | 41.9% | 9.7% |
| Portugal NBSAP (2026) | 549 | 43.4% | 10.6% | 40.8% | **13.8%** |
| Australia LT-LEDS3 (Nov 2025) | 2,697 | 29.1% | 8.6% | 27.1% | 4.6% |
| Türkiye LT-LEDS (Nov 2024) | 463 | 27.9% | 9.3% | 26.3% | 3.7% |
| Armenia NBSAP (2026) | 55 | 27.3% | 12.7% | 20.0% | 7.3% |
| Uganda NBSAP (2026) | 402 | 25.1% | 11.9% | 21.9% | 7.7% |

The clearest pattern: **for both countries the headline pledge is far denser in
justice language than the long-term strategy behind it.** Türkiye's NDC 3.0 runs
51.9% against its LT-LEDS at 27.9%; Australia's 2035 NDC 45.2% against LT-LEDS3
at 29.1%. The same holds on every individual classifier, not just the total.

Don't over-read it as rhetoric-versus-substance, though. ANDCs are short
political documents (54 and 155 passages here) and LT-LEDS are long technical
ones (463 and 2,697) carrying inventory tables and sectoral detail that dilute
any single theme. The gap is real; the causal story is not settled by these
numbers.

Among these three NBSAPs Portugal looks procedurally heavy — 43.4% overall and
13.8% procedural against Uganda (25.1% / 7.7%) and Armenia (27.3% / 7.3%) — but
that is an artefact of the comparison set. Across all 114 NBSAPs with at least
100 passages, procedural justice runs from 0.9% to 28.7% with a median of 8.7%.
Portugal's 13.8% ranks 30th: above median, nowhere near the top (Cambodia 28.7%,
Tuvalu 25.9%, Timor-Leste 24.8%). Uganda at 7.7% is the unusual one, ranking
72nd, below the median.

The within-country trend is the more interesting number: Portugal's own 2018
NBSAP scored 9.1% (rank 53), so the 2026 revision raised its procedural share by
half again.

Procedural justice is the thinnest strand in every document in figs 5a-5b except
Türkiye's NDC 3.0 and Portugal's NBSAP.

Türkiye's NDC 3.0 is only 54 passages, so one passage moves its rate by 1.9
points; read the short rows as patterns, not rates.

**Laws (fig 6).** 63 published laws across the two countries; 10 have no
passages, 7 more fall under a 30-passage floor, leaving 29 Australian and 17
Turkish. Türkiye's pooled justice density is 14.2% against Australia's 3.3%.
Top: Act No. 5403 on Soil Conservation and Land Use (35.9%), Law 7552 Climate
Law (30.4%), Eleventh National Development Plan (27.5%).

Composition explains much of that gap and should be stated if published:
Türkiye's "Law" set includes a national development plan, while Australia's
largest law by volume is the Higher Education Support Act (3,526 passages,
4.1%). Comparing countries on category membership compares legal drafting
convention as much as policy substance.

**What the justice language says in each country (fig 7).** Run separately
within each justice type, because pooled it mostly reproduced the distributive
class, which is three times the size of the others:

| | Australia | Türkiye |
|---|---|---|
| **Climate justice** (2,649 / 1,894) | government, communities, nations, community, energy, million · *net zero, clean energy, developing countries* | women, social, children, development, institutions, child, education · *action plan, change adaptation* |
| **Distributive justice** (6,393 / 3,736) | government, emissions, communities, net, million, program · *net zero, clean energy, emissions reduction* | agricultural, water, green, development, social, order · *action plan, climate change, water resources* |
| **Procedural justice** (1,516 / 690) | government, communities, community, nations, program, indigenous, state, traditional · *torres strait, state territory* | participation, institutions, women, public, process, social · *action plan, climate change* |

The split earns its keep in the procedural panel, which the pooled version hid:
Australian procedural justice is about First Nations and the federal-state
relationship (`indigenous`, `traditional`, `torres strait`, `state territory`),
while Türkiye's is about participation and institutions. Q912 is also the
thinnest panel — 690 Turkish passages — so treat it as the most provisional.

Turkish documents here are largely machine-translated, so part of this
separation is translation register rather than policy substance; `order` on the
Turkish distributive side is the clearest example.

## 6. What each classifier travels with (fig 8)

Co-occurrence against the two concept families, measured as lift —
P(concept | justice) / P(concept) — so a common concept does not simply
dominate every column. Families taken from the concept store hierarchy; only
members carrying a primary classifier reach `topics`, giving 6 of Q47's
descendants and 9 of Q672's. Q672 itself has no classifier.

| Concept | Q32 | Q911 | Q912 |
|---|---:|---:|---:|
| legal safeguards for vulnerable groups | 7.6× | 3.4× | **12.0×** |
| social inclusion | 6.1× | 2.9× | **10.2×** |
| indigenous people | 6.4× | 3.3× | **10.0×** |
| marginalized ethnicity | 6.4× | 3.3× | 9.7× |
| sexual minority | 7.2× | 2.8× | 6.2× |
| women and minority genders | 7.2× | 3.0× | 5.6× |
| just transition | 4.9× | 2.7× | 5.6× |
| people with limited assets | 5.0× | 3.5× | 3.6× |
| green jobs | 2.1× | 2.0× | 2.0× |

Every value exceeds 1×, as expected for justice-adjacent concepts. What differs
is the spread. **Procedural justice has both the highest peaks and the widest
range** (1.9× to 12.0×) — it is strongly enriched for the language of who gets
a seat at the table. **Climate justice is uniformly high** (2.1× to 7.6×),
consistent with an umbrella. **Distributive justice is flat at 2.0–3.5× against
everything**, the same weak specificity its own vocabulary showed in §2 — it is
not preferentially attached to any impacted group or just-transition theme.

Caveat: both Q911 and Q912 are formally `subconcept_of` Q32 in the concept
store, and Q47 lists Q32 as a related concept, so part of this structure is
ontology design rather than independent discourse.

## 7. Over time (figs 9 and 10)

The raw share of the corpus carrying a justice label peaks at 33.1% in 2016 —
the Paris Agreement year — holds near a third through 2022, then drops sharply
to 22.9% in 2023 and sits around a quarter since.

**That drop is one specific thing: national greenhouse gas inventory reports.**
They are emissions accounting — page after page of source-category tables — and
they carry justice language on about 1% of passages. They enter the corpus at
scale in 2023:

| | 2022 | 2023 | 2025 |
|---|---:|---:|---:|
| emissions inventories, passages | 11,967 | **87,486** | 87,003 |
| emissions inventories, justice rate | 1.3% | 1.0% | 1.2% |

A 7× jump in one year, at a rate near zero, is enough to pull the whole corpus
down eight points. Split them out and every other corpus is flat or rising:

| corpus | 2016 | 2022 | 2025 |
|---|---:|---:|---:|
| Law + Policy | 33.4% | 33.2% | 30.9% |
| UN submission, excluding inventories | 28.3% | 29.0% | **40.9%** |
| MCF project | 36.3% | 47.3% | **55.3%** |
| Emissions inventories | 4.1% | 1.3% | 1.2% |

UN submissions are the striking one. Pooled with inventory reporting they look
like a corpus losing interest in justice; separated, they are the fastest-rising
of the three substantive corpora, from 28.3% to 40.9%.

These reports are kept in the figure rather than filtered out, shown as a
second UN submission panel so the parent-child relation is visible, and the
small panels carry a pale band for total volume — without it the panel plots
only the ~1% that is labelled and the 87,000 passages doing the diluting are
invisible. The point is not that the corpus is contaminated — it is that a share
computed over everything CPR ingests is not a measure of policy discourse, and
the composition has changed faster than the discourse has.

Corporate disclosure is excluded from this series for a different reason: that
dataset has not been refreshed this year, so all its passages sit in the past
and would bend a trend rather than inform it.

### Distributive and procedural, without the umbrella (fig 10)

The "more than one label" band in fig 9 hides how the two subconcepts move
together, so fig 10 re-cuts the same data as an independent cross-tab of Q911
and Q912, ignoring Q32 entirely.

| | 2016 | 2021 | 2025 |
|---|---:|---:|---:|
| distributive only | 22,307 | 51,542 | 46,283 |
| both | 6,695 | 10,006 | **14,224** |
| procedural only | 3,718 | 4,811 | 4,154 |

**Procedural justice barely grows on its own.** Passages carrying only
procedural language are essentially flat across a decade, from 3,718 to 4,154,
while passages carrying both more than double. As a share of all procedural
passages, those co-occurring with distributive language rise from 64% to 77%.

Procedural justice is increasingly something policy text does *while* making a
distributive claim, rather than a separate register of its own — consistent
with §6, where procedural justice is also the classifier most enriched for
named impacted groups.

---

## Data notes worth passing to the data team

1. **`DOCUMENTS.passage_count` is rolled up from the v1 passages table.** For
   the deep-dive documents it matches `PASSAGES_V1` row counts *exactly* and
   overstates the real `PASSAGES` count 7–11× (Uganda 4,996 vs 441; Armenia
   1,201 vs 58; Türkiye NDC1 671 vs 109; Australia 2022 NDC 394 vs 27). It is
   unusable as a denominator — this looks like a second v1 dependency in the
   canonical models beyond the documented `labels[].passages_id` one.
2. **`geographies` values are prefixed** (`country::UGA`, `region::ECS`,
   `subdivision::US-RI`), and **`countries` holds country *names*, not ISO-3**
   ("Liechtenstein"); `regions` likewise ("Sub-Saharan Africa"). Filtering
   `array_contains('UGA', geographies)` silently returns nothing.
