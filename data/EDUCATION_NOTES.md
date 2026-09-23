# Education Expenditure Dataset — Codebook and Data Quality Notes

**Source:** Chatterjee and Hadi, *Regression Analysis by Example* (5th ed.), the education
expenditure data used for the qualitative-predictor and stability-over-time examples in
Chapter 5. Distributed with the book as data file `P151` (RABE names its data files by
page number). The table on the slides is cited as Table 5.12. The figures are the **1960**
cross-section: per-capita incomes of \$1,053–\$2,817 are 1960 dollars (the same chapter's
1970 and 1975 cross-sections are not in this file).

**File:** `education.csv` — 50 rows × 6 columns. One row = one state. Built from
`P151.txt` on 2026-09-23: whitespace stripped, RABE's `STATE Y X1 X2 X3 Region` renamed
to plain names, **row order preserved** (index 0 = ME … 49 = HI).

**Encoding:** plain ASCII. No `encoding=` argument needed:

```python
edu = pd.read_csv(edu_url)
```

**Codebook:** `education_codebook.csv` — all 6 columns, same layout as
`carnegie_codebook.csv` (label, source, dtype, missing count, unique count, range, value
labels).

**Raw URL:** `https://raw.githubusercontent.com/ryanaxiom/Applied-ML/main/data/education.csv`
(verified HTTP 200 on 2026-09-23; byte-identical to the built file).

---

## Units

| Column | What it measures | Units |
|---|---|---|
| `spend` | per-capita expenditure on public education (**the target**) | dollars per resident, 1960 |
| `income` | per-capita personal income | dollars per resident, 1960 |
| `young` | residents under 18 | **per 1,000** residents |
| `urban` | residents living in urban areas | **per 1,000** residents |

So `young = 388` means 38.8% of Maine's population was under 18, and `urban = 899` means
89.9% of Rhode Island's was urban. Mixed units on one table — dollars beside
per-thousand rates — is the first thing to say when the file goes on screen.

**Naïve model:** mean `spend` **85.04**, median 81.0, RMSE **20.72**
(`= spend.std(ddof=0)`, exactly as with Carnegie). Every Day 12 improvement is quoted
against 20.72.

---

## `region` is a code, not a quantity

`region` is stored as an integer and carries no numeric meaning — the same trap as
Carnegie's `locale` and `control`.

| Code | Region | States | Mean `spend` |
|---|---|---|---|
| 1 | Northeast | 9 | 77.7 |
| 2 | North Central | 12 | 87.2 |
| 3 | South | 16 | 70.5 |
| 4 | West | 13 | 106.0 |

The codes follow the Census regions (Delaware and Maryland are South). The West spends
the most per resident, the South the least, and the ordering 1–4 has nothing to do with
that. If `region` is ever used as a predictor it is one-hot encoded (Day 8), never fed in
as a number. **Day 12 does not use it**; it is in the file because RABE's own example
compares the regression across regions.

---

## Two states carry one variable

`young` has a long right tail with exactly two states in it:

| State | `young` | `spend` |
|---|---|---|
| Alabama | **637** | 59 |
| Kentucky | **594** | 49 |
| Utah (next) | 494 | 109 |
| New Mexico | 458 | 94 |

Alabama and Kentucky are 100–143 per-thousand above the next state, and both spend little,
so **the overall correlation of `young` with `spend` is −0.21** — while `income` is +0.65
and `urban` is +0.25. Everything a linear model learns about high-`young` states comes
from those two rows. This is the mechanism behind Day 12: whether AL and KY share a test
fold decides whether 5-fold CV reports the model as a 20% improvement (apart: RMSE
15.57–18.28) or as worse than the mean (together: 20.27–24.19), with no overlap between
the two ranges over 200 shuffles.

Kentucky is also the minimum of the target (`spend` 49); Wyoming is the maximum (142).
Mississippi has the lowest income (1,053), Connecticut the highest (2,817).

---

## One quirk kept from the source

Row 19 is **`NB`**, which is Nebraska. `NB` was Nebraska's postal abbreviation from 1963
until November 1969, when it became `NE` at Canada's request (New Brunswick). RABE kept
the old code and so does this file, so that the data matches the book. A student who
notices has read the data; the answer is "that is Nebraska, and the book is older than
the fix."

Nothing in the course code depends on it. If it ever needs to join another table, map it:

```python
edu["state"] = edu["state"].replace({"NB": "NE"})
```

---

## No missing values, no sentinels

Every cell is populated and every value is plausible. There is nothing here like
Carnegie's `locale = -3` or the SAT zeros — which is worth saying out loud, because the
Day 12 lesson is that a clean dataset can still produce an unstable estimate. The
instability is not a data-quality problem.

---

## Structural notes

- **Index order matters.** The Day 12 slides locate Alabama and Kentucky by position
  (`edu.index[edu["state"].isin(["AL", "KY"])]` → 31 and 29) and pass those positions to
  `KFold.split`. Do not sort, filter, or re-save the file with a different row order.
- **One row per state, no District of Columbia**, no territories. `n = 50` and every
  5-fold test fold has exactly 10 rows — the "training rows per parameter" table on Day 12
  (40 rows, 4 parameters) rests on this.
- **Rows are independent.** One cross-section, one year, no nesting, so standard
  cross-validation is appropriate. What is *not* appropriate at this size is trusting one
  shuffle: report the spread across repeated shuffles (Day 12, slide 12).
- **Second dataset, same questions.** It exists in the course so the Carnegie results
  (n = 302) have a small-n control. Anything demonstrated on Carnegie can be rerun here
  in one line by swapping `d[pile], y` for `three, spend`.

---

## Verified reference numbers (seed 309 unless stated)

| Quantity | Value |
|---|---|
| naïve RMSE | 20.72 |
| 5-fold CV RMSE, `income + young + urban` | 16.62 (seeds 0 / 1 / 2: 16.90 / **21.85** / 16.79) |
| fold RMSEs, seed 309 | 21.33 / 16.16 / 19.31 / 11.02 / 13.09 |
| 200-shuffle range | 15.57 – 24.19, 51.0% of the median; 39 of 200 worse than naïve |
| three-input vs two-input (`income + young`) | three wins 120 of 200; median paired difference −0.08 |
| AL & KY in the same test fold | 42 of 200 shuffles, RMSE 20.27 – 24.19 |
| AL & KY in different folds | 158 of 200, RMSE 15.57 – 18.28 |
| mean of 10 shuffles | 17.30; spread of the average 51.0 → 14.4 → 5.7% at R = 1, 10, 20 |
| `RepeatedKFold(5, 20)`, `cross_val_score` route | 17.45 |

Full derivations are in the Day 12 instructor notes and `Day12_Carnegie.ipynb`.

---

*Compiled for DSCI 309: Applied Machine Learning. All figures verified directly against
`education.csv` as built from RABE `P151.txt`.*
