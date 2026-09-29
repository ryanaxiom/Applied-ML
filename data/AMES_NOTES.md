# Ames Housing Dataset — Codebook and Data Quality Notes

**Source:** De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data as an
End of Semester Regression Project." *Journal of Statistics Education* 19(3). Every
residential sale in Ames, Iowa from 2006 through 2010, assembled from the Ames City
Assessor's records. The author's variable documentation is
`https://jse.amstat.org/v19n3/decock/DataDocumentation.txt`; every label and level code in
the codebook below was transcribed from it.

**File:** `ames_housing.csv` — 2,930 rows × 82 columns. One row = one sale of one parcel.
Built on 2026-09-22 from De Cock's tab-separated `AmesHousing.txt`: converted to CSV with
**column names, row order, and every cell unchanged**, including a trailing space in one
level code (see below). The five houses over 4,000 sq ft that De Cock tells instructors to
remove are **still in the file** — deliberately.

**Encoding:** plain ASCII. No `encoding=` argument needed. But *how* it is read matters
more than for any other file in the course (see "NA is a level, not a hole"):

```python
ames = pd.read_csv(ames_url)                                        # the default read
ames = pd.read_csv(ames_url, keep_default_na=False, na_values=[""])  # the honest read
```

**Codebook:** `ames_codebook.csv` — all 82 columns, same layout as `carnegie_codebook.csv`
and `education_codebook.csv` (variable, label, source, dtype, missing count, unique count,
range, value labels). Missing counts are from the **default** read, since that is what a
student sees first; the value labels say which `NA` codes the reader turned into NaN.

**Raw URL:** `https://raw.githubusercontent.com/ryanaxiom/Applied-ML/main/data/ames_housing.csv`
(not yet pushed as of 2026-09-24; `education.csv` at the same path returns 200).

---

## Units

| Column group | Units |
|---|---|
| `SalePrice` (**the target**) | dollars, at the time of sale |
| `Lot Area`, all `* SF` columns, `Gr Liv Area`, `Garage Area`, the porch and deck columns, `Pool Area` | square feet |
| `Lot Frontage` | linear feet of street |
| `Year Built`, `Year Remod/Add`, `Garage Yr Blt`, `Yr Sold` | calendar year |
| `Mo Sold` | month number 1–12 (a code, not a quantity) |
| `* Bath`, `Bedroom AbvGr`, `Kitchen AbvGr`, `TotRms AbvGrd`, `Fireplaces`, `Garage Cars` | counts |
| `Overall Qual`, `Overall Cond` | assessor's rating 1–10, ordinal |
| the `Ex / Gd / TA / Fa / Po` columns | assessor's rating, ordinal, stored as text |
| `Misc Val` | dollars |

**Naïve model:** mean `SalePrice` **$180,796**, median **$160,000**, RMSE **$79,873**
(`= SalePrice.std(ddof=0)`, exactly as with Carnegie). Every Project 1 improvement is
quoted against $79,873. The target is right-skewed (skew +1.74, range $12,789–$755,000),
so the Day 9 log-target discussion applies.

---

## NA is a level, not a hole

This is the file's defining feature and the first thing to say about it.

De Cock uses **two different tokens** for two different things, and his documentation says
so: the literal text `NA` means *the house does not have this* (no garage, no basement, no
pool, no fence, no alley, no fireplace, no miscellaneous feature), while a **blank cell**
means *the value was not recorded*. pandas' default reader treats both as missing. The
result:

| Read | NaN cells in the file |
|---|---|
| `pd.read_csv(ames_url)` | **15,749** |
| `pd.read_csv(ames_url, keep_default_na=False, na_values=[""])` | **719** |

**Ninety-five percent of the apparent missingness is the word "none."** A student who
runs `dropna()` on the default read keeps a few dozen rows and has thrown away nearly the
whole dataset for no reason.

The proof is one line, because the "none" columns have numeric partners that agree with
them exactly under the default read:

| Text column, NaN count | Numeric partner | Count equal to zero |
|---|---|---|
| `Garage Type` 157 | `Garage Cars` | **157** |
| `Pool QC` 2,917 | `Pool Area` | **2,917** |
| `Fireplace Qu` 1,422 | `Fireplaces` | **1,422** |
| `Misc Feature` 2,824 | `Misc Val` | 2,827 (three features valued at $0) |
| `Bsmt Qual` 80 | `Total Bsmt SF` | **79** — and one blank |

The basement row that does not match is **PID 903230120** (row 1341): every basement
column is blank, including the square-footage columns. That is not a house with no
basement; it is a house whose basement was never recorded. Under the honest read it is
the file's one basement NaN.

**Sixteen text columns use `NA` as a documented level:** `Alley`, `Bsmt Qual`,
`Bsmt Cond`, `Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2`, `Fireplace Qu`,
`Garage Type`, `Garage Finish`, `Garage Qual`, `Garage Cond`, `Pool QC`, `Fence`,
`Misc Feature` — plus `Mas Vnr Type`, which is its own case:

**`Mas Vnr Type` and the word `None`.** De Cock's code for "no masonry veneer" is the
literal text `None` (1,752 houses). The string `None` is on pandas' default NA list, so
the default read converts it to NaN and merges it with the **23** genuinely blank rows.
Under the default read the column shows 1,775 NaN and four levels; under the honest read
it shows 23 NaN and five levels, `None` being the most common. The data is fine. The
reader changed it.

**What is genuinely missing (honest read, 719 cells):** `Lot Frontage` 490 (16.7% — the
only column with real, substantial missingness), `Mas Vnr Type` / `Mas Vnr Area` 23 each,
`Garage Yr Blt` 159 (the 157 no-garage houses plus two detached garages with no year),
and a scattering of ones and twos in the basement, garage, and `Electrical` columns.

**PID 910201180** (row 2236) is the garage analogue of the basement row: `Garage Type` is
`Detchd`, and year, finish, cars, area, quality, and condition are all blank. A detached
garage that was never measured. It is the one NaN in `Garage Cars` and `Garage Area`.

---

## Codes stored as numbers

- **`MS SubClass`** — the type of dwelling (`20` = 1-story 1946 & newer, `30` = 1-story
  1945 & older, `60` = 2-story 1946 & newer, `120` = 1-story PUD, `190` = 2-family
  conversion, and so on: 16 codes). Stored as an integer; **means nothing as a number.**
  This is Carnegie's `locale` and the education file's `region`, one more time. A student
  who feeds it to `LinearRegression` as-is has fit a slope to a zip-code-like label.
- **`Mo Sold`** — month of sale, 1–12. December is not eleven units of anything more than
  January.
- **`PID`** and **`Order`** — identifiers. `PID` is a parcel number usable on the city's
  website; `Order` is the row number. Neither is a predictor.
- **`Overall Qual`** and **`Overall Cond`** — 1–10 ratings. These *are* ordered, so using
  them as numbers is defensible and common. Note `Overall Cond` never reaches 10 in the
  data (max 9).
- **The `Ex / Gd / TA / Fa / Po` columns** (`Exter Qual`, `Kitchen Qual`, `Heating QC`,
  `Bsmt Qual`, `Garage Qual`, …) are ordinal ratings stored as **text**. One-hot encoding
  works and discards the order; mapping to `5/4/3/2/1` keeps the order and assumes equal
  spacing. Either is a modeling decision worth one sentence in a memo.

---

## Columns that add up exactly

Two identities hold on every row, to the square foot:

```
BsmtFin SF 1 + BsmtFin SF 2 + Bsmt Unf SF  ==  Total Bsmt SF
1st Flr SF + 2nd Flr SF + Low Qual Fin SF  ==  Gr Liv Area
```

Put all four terms of either identity in one regression and you have Day 7's collinearity
at r = 1 — the dummy trap in square feet. The coefficients will be arbitrary and the
predictions will be fine, which is exactly the "twins tax interpretation, not prediction"
lesson, and a good thing for a student to discover rather than be told.

---

## Impossible and suspicious values

Verified against the file. None of these is documented by De Cock.

- **`Garage Yr Blt = 2207`** on PID 916384070 (row 2260): house built 2006, sold 2007,
  garage built two centuries from now. Almost certainly `2007`. It is the only value above
  2010 and the cleanest impossible-value find in the file.
- **Remodeled before it was built:** PID 907194160 (row 850) has `Year Built` 2002 and
  `Year Remod/Add` 2001. One row.
- **Sold before it was built:** one house has `Yr Sold` earlier than `Year Built`, and
  three have `Yr Sold` earlier than `Year Remod/Add`. Pre-sales of new construction are a
  real thing (see `Sale Condition = Partial`), so these are "worth a sentence," not
  errors.
- **`Sale Type` has a trailing space.** The dominant level is stored as `'WD '` — `W`,
  `D`, space — on 2,536 rows. `ames["Sale Type"] == "WD"` matches **zero rows** with no
  error. This is the Day 2 `"wv"` vs `"WV"` trap in a new form; `.str.strip()` fixes it.
- **Levels present differ from the documented spelling** in `MS Zoning`: the file has
  `C (all)`, `I (all)`, `A (agr)` where the documentation lists `C`, `I`, `A`. `RP` is
  documented but absent. Filtering on the documented code finds nothing.
- **Houses with zero of something they should have:** 8 with no bedroom above grade, 3
  with no kitchen above grade, 12 with no full bath above grade. Some are real (a
  basement-bedroom bungalow), some may not be. Not errors on their face; worth a look.
- **Near-constant columns:** `Utilities` is `AllPub` on 2,927 of 2,930 rows; `Street` is
  paved on 2,918; `Roof Matl` is composite shingle on 2,887. They carry no information for
  prediction and will produce one-hot columns of almost all zeros.

---

## Five houses over 4,000 square feet

De Cock's documentation ends with an instruction to instructors: remove the five houses
with `Gr Liv Area` above 4,000 before giving the data to students. A plot of `SalePrice`
against `Gr Liv Area` shows why:

| PID | `Gr Liv Area` | `SalePrice` | `Sale Condition` | Neighborhood |
|---|---|---|---|---|
| 908154235 | 5,642 | $160,000 | Partial | Edwards |
| 908154195 | 5,095 | $183,850 | Partial | Edwards |
| 908154205 | 4,676 | $184,750 | Partial | Edwards |
| 528320050 | 4,476 | $745,000 | Abnorml | NoRidge |
| 528351010 | 4,316 | $755,000 | Normal | NoRidge |

Three are `Partial` sales in the same neighborhood — houses that were not finished when
assessed, sold for a third of what their size implies. Two are simply very large,
expensive houses priced about right. **They are left in the file on purpose.** They are
Day 6's outlier-versus-leverage lesson (Johns Hopkins, Central Florida) appearing in the
wild, and the decision about what to do with them — drop, keep, model separately, restrict
to `Sale Condition == "Normal"` — is a modeling decision a student should make and
defend, not one made for them upstream. De Cock's own caution applies: do not discard data
because it disagrees with you; discard it because you can say what it is.

---

## Sales that are not market sales

`Sale Condition` is the column that says whether a price means what it looks like:

| Level | Count | What it is |
|---|---|---|
| `Normal` | 2,413 | ordinary arm's-length sale |
| `Partial` | 245 | house not complete when last assessed (new construction) |
| `Abnorml` | 190 | trade, foreclosure, or short sale |
| `Family` | 46 | sale between family members |
| `Alloca` | 24 | two linked properties with separate deeds (condo + garage unit) |
| `AdjLand` | 12 | adjoining land purchase |

517 of 2,930 sales, 17.6%, are not `Normal`. Whether to model all of them, only the
`Normal` ones, or all of them with `Sale Condition` as a predictor changes what the model
is *for* — predicting assessments, or predicting what a buyer pays — and is worth a
paragraph in a memo.

---

## Column names differ from the documentation

The file's headers were set by whoever converted the assessor's export, not by De Cock's
write-up, so nine names differ from the documentation:

| In the file | In the documentation |
|---|---|
| `Exterior 1st`, `Exterior 2nd` | `Exterior 1`, `Exterior 2` |
| `BsmtFin Type 2` | `BsmtFinType 2` |
| `Heating QC` | `HeatingQC` |
| `Bedroom AbvGr`, `Kitchen AbvGr` | `Bedroom`, `Kitchen` |
| `Kitchen Qual` | `KitchenQual` |
| `TotRms AbvGrd` | `TotRmsAbvGrd` |
| `Fireplace Qu` | `FireplaceQu` |
| `3Ssn Porch` | `3-Ssn Porch` |

The codebook uses the file's names and notes the documented one. **Every name contains a
space, and one contains a slash** (`Year Remod/Add`), so column access is
`ames["Gr Liv Area"]`, never `ames.Gr_Liv_Area`, and `patsy`/`statsmodels` formulas need
`Q("Gr Liv Area")`. This is inconvenient and it is left alone, because it keeps the file
matching the documentation students are told to cite.

---

## Structural notes

- **Rows are independent.** `PID` is unique — no parcel appears twice — so there are no
  repeat sales and no panel structure. The five sale years are pooled as one
  cross-section; `Yr Sold` is available as a predictor if a student wants to allow for
  the 2008 market.
- **No duplicate rows.** Checked.
- **Row order does not matter** for anything in the course (unlike `education.csv`, where
  the Day 12 slides locate Alabama and Kentucky by position). Sorting is safe.
- **39 numeric and 43 text columns** under the default read. The 43 are where most of the
  file's information is; a model that uses only the numeric columns is leaving the
  majority of the data on the table, and Project 1 says so.
- **Second dataset, different job.** `education.csv` exists as a small-n control for
  Carnegie; `ames_housing.csv` exists as the Project 1 assignment — a dataset the students
  have never seen, rich enough that the diagnosis is most of the work.

---

## Verified reference numbers

All from the default read, `KFold(n_splits=5, shuffle=True, random_state=309)`,
`LinearRegression`.

| Quantity | Value |
|---|---|
| naïve RMSE | $79,873 |
| `SalePrice` mean / median / skew | $180,796 / $160,000 / +1.74 |
| NaN cells, default read / honest read | 15,749 / 719 |
| strongest single correlations with `SalePrice` | `Overall Qual` 0.799 · `Gr Liv Area` 0.707 · `Garage Cars` 0.648 · `Garage Area` 0.640 · `Total Bsmt SF` 0.632 |
| grading reference: `Overall Qual`, `Gr Liv Area`, `Total Bsmt SF`, `Garage Area`, `Year Built` (2,928 rows after dropping the 2 NaN) | 5-fold CV RMSE **$37,015**, 53.7% better than naïve |
| houses over 4,000 sq ft | 5 (three `Partial`) |
| non-`Normal` sales | 517 of 2,930 (17.6%) |

Full derivations are in the Project 1 materials.

---

*Compiled for DSCI 309: Applied Machine Learning. All figures verified directly against
`ames_housing.csv` as built from De Cock's `AmesHousing.txt`.*
