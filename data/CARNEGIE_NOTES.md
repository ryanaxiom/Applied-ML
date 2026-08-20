# Carnegie Dataset — Codebook and Data Quality Notes

**Source:** Carnegie Classification of Institutions of Higher Education, 2021 edition
(CCIHE2021). Variable labels, sources, and value codes extracted from the `Variable` and
`Values` sheets of `CCIHE2021PublicData.xlsx`.

**File:** `carnegie_data.csv` — 3,939 rows × 101 columns. One row = one institution.

**Encoding:** `cp1252`, not UTF-8. Load with:

```python
carnegie = pd.read_csv(url, encoding="cp1252")
```

**Codebook:** `carnegie_codebook.csv` — all 101 columns documented, with variable label,
originating source (IPEDS / NSF / CCIHE / other), dtype, missing count, unique count,
range, and value labels where the variable is categorical.

---

## Units

`serd` and `nonserd` are **research and development expenditures in thousands of dollars**,
sourced from NSF. A value of `1000000` means one billion dollars.

Sanity check: Johns Hopkins is the largest at `3110494`, or $3.11B, which matches its
published research expenditure.

---

## Two different binary conventions

The file does not use one convention. This is documented in the source codebook, but it is
easy to miss and easy to get wrong.

| Coding | Variables |
|---|---|
| `1 = Yes`, `2 = No` | `medical`, `hbcu`, `tribal`, `landgrnt` |
| `0 = No`, `1 = Yes` | `msi`, `hsi`, `womens`, `docresflag` |

Consequence: `carnegie["medical"].mean()` returns **1.95**, which is not a proportion.
The actual share of institutions granting a medical degree is **5.1%** (199 of 3,939).

Recode before using any of the `1/2` variables as a target or interpreting a coefficient:

```python
carnegie["medical_yn"] = (carnegie["medical"] == 1).astype(int)
```

---

## Nominal codes that are not quantities

Several variables are stored as integers but carry no numeric meaning. Computing means,
or feeding them to a distance-based method such as k-means or KNN without encoding, produces
results that look fine and mean nothing.

- **`locale`** — urban-centric locale. `11/12/13` = City (Large/Midsize/Small),
  `21/22/23` = Suburb, `31/32/33` = Town (Fringe/Distant/Remote), `41/42/43` = Rural.
  Rural Remote (43) is not "32 units away from" City Large (11).
- **`control`** — `1` = Public, `2` = Private not-for-profit, `3` = Private for-profit.
- **`obereg`** — geographic region code.
- **`basic2000`–`basic2021`** — Carnegie Basic classification codes.

`selindex` (`1` = inclusive, `2` = selective, `3` = more selective) is **ordinal**, which is
a third case: the ordering is real but the spacing is not.

---

## Documented missing-value sentinels

- **`locale = -3`** — documented in the source codebook as `{Not available}`. Three
  institutions: College of Micronesia-FSM, College of the Marshall Islands, and Palau
  Community College.
- **`basic2000` through `basic2018` = -2** — documented as "Not classified, not in
  classification universe." Affects 145–898 rows depending on the year.

These are valid integers. No software will warn you about them.

---

## Undocumented values found in the data

Three values appear in the data that the source codebook does not define.

1. **`locale = 25`** — one institution, Uniformed Services University of the Health
   Sciences (unitid 164137). **There is no locale code 25** in the IPEDS scheme; valid
   suburb codes are 21, 22, and 23. This looks like a genuine data error. Treat as missing.

2. **`womens = 2`** — one institution, University of Denver (unitid 127060). The codebook
   documents only `0 = No` and `1 = Yes`. Denver is not a women's college. Treat as
   missing or as `0`.

3. **`hsi = 0`** — 3,607 rows. Benign: the codebook documents only the positive category,
   and `0` plainly means "not an HSI."

---

## Undocumented sentinel: `satcmb25`

`satcmb25` (combined SAT Math + Verbal, 25th percentile) contains **57 zeros among the 302
institutions reporting research expenditure** (18.9%). The observed nonzero range is
840–1530.

A combined SAT score of zero is not possible, and these are almost certainly "not reported"
— many institutions went test-optional. **The source codebook does not document this**, so
it is an inference, not a documented code. State it that way.

`pdnfrstaff` (postdoctoral staff) similarly has 26 zeros, though zero is a plausible true
value there.

---

## Structural notes

- **Branch campuses.** Some universities appear as several rows (West Virginia University
  has 4, University of Pittsburgh has 5). Typically only the main campus row carries
  research expenditure; branch rows are missing. Check `unitid` before assuming one row per
  university.
- **Research expenditure coverage.** Only **302 of 3,939** institutions report `serd` and
  `nonserd`. Because the two are summed, a missing value in either produces a missing total
  — the 302 is an intersection, not a union.
- **Rows are independent.** Institutions are observed once, in one year. There is no time
  dimension and no nesting, so standard cross-validation is appropriate for this file.

---

*Compiled for DSCI 309: Applied Machine Learning. All figures verified directly against
`carnegie_data.csv` and the CCIHE2021 source codebook.*
