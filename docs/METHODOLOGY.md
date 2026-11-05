# Methodology

## Research question

Among Israeli inventors, is a patent by an Arab inventor less likely to be produced
inside a firm the inventor does not own than a patent by a Jewish inventor with the same
entry cohort, technology field and region? This is the intrapreneurship question:
invention inside the firm versus outside it.

Because every inventor in the data already holds a patent, the study cannot see people
who never reached that point. The design therefore asks a second question the data can
answer well: once Arab engineers are inside firms, who do they collaborate with?

## Data

PatentsView *Granted Patent Disambiguated* bulk files from the USPTO Open Data Portal:
inventor locations, disambiguated inventors, applications, disambiguated assignees and
current CPC classifications. The full history is read (1976 onward) so that an inventor's
first patent is identified correctly; analysis cohorts are inventors whose first patent
was filed in 1990–2019.

## Sample construction

| Step | Rule |
|---|---|
| Inventor | Disambiguated `inventor_id` resident in Israel on at least one patent |
| Exposure | Exactly five filing years from each inventor's first patent, so careers of different length are comparable |
| Cohorts | First patent 1990–2019 (left-truncation avoided by reading the full history) |
| Ownership | Each patent is *corporate* (assigned to a firm the inventor does not own), *independent* (unassigned or inventor-owned) or excluded |
| Exclusions | University and government assignees, removed by assignee type code **and** by a name list of technology-transfer companies registered as ordinary firms |
| Founders | Inventor-owned firms identified by an explicit, reproducible rule rather than judgement |
| Analysis set | Inventors with a clear Arab or clear Jewish surname; the ambiguous bucket is counted and reported, then excluded |

The flow from 48,111 raw inventors to 17,641 in the analysis sample is in
`results/flow.csv` and `figures/fig8_sample_funnel.png`.

## Measurement

**Surname classifier.** A curated lexicon assigns each surname to Arab, Jewish or
ambiguous, with explicit handling of names shared across communities. The ambiguous bucket
is deliberately large: misclassifying a hidden Arab inventor as Jewish would contaminate
the comparison group, while excluding costs only precision.

**Validation.** A sample stratified by *predicted* class is labelled by hand by two
independent raters. Precision is estimated directly; recall is not, because true Arab
inventors in the other buckets are under-sampled. The script estimates
P(true | predicted) from the sample and reweights by the real size of each predicted
class, then reports precision, recall, F1 and Cohen's kappa.

**Region.** Cities map to districts and to an Arab-majority-locality flag. Localities
where the USPTO city field was found to be systematically wrong are excluded from the
locality measure.

## Models

Let *C* be corporate patents and *T* total patents in the five-year window.

| Model | Outcome | Role |
|---|---|---|
| Binomial GLM, *T* trials | corporate share *C/T* | Main specification |
| Logit | any corporate patent | Extensive margin |
| PPML | corporate count *C* | Output |
| Firm fixed effects | within-firm comparison | Separate question: does the gap hold inside the same employer? |

Each model is estimated as a ladder (cohort only → + region → + field → full). Region and
field may themselves be shaped by discrimination, so the movement of the Arab coefficient
across columns is read as evidence, not any single column.

## Co-inventor homophily

The share of Arab co-inventors on Arab inventors' patents is compared with two permutation
nulls:

- **Global null.** Ethnicity labels shuffled across all inventors. Exceeding it shows only
  that clustering exists.
- **Within-firm null.** Labels shuffled only among inventors of the same firm, preserving
  each firm's ethnic composition exactly. Exceeding it is within-firm segregation and
  cannot be explained by sorting across employers.

## Robustness

- Three ethnicity definitions with different error structures: surname, Arab-majority
  locality, and their union. Agreement across all three is stronger evidence than any one.
- Balance and sensitivity checks on the excluded ambiguous names, including bounds from
  reassigning them.
- A name-free count of Arab inventors among excluded records, using reliable localities
  only, to bound the true Arab share.

## Verification

- `make_synthetic.py` plants a known effect (odds ratio 0.55); the pipeline must recover a
  negative, significant coefficient.
- `selftest.py` runs 30 invariant checks, including recomputing patent totals from raw
  records for sampled inventors.
- Both run in CI on every push.
