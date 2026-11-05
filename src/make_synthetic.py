"""
Synthetic raw_records.csv with a KNOWN true effect, so we can verify the
pipeline recovers it before touching real data. Not for the paper.

Truth planted: Arab inventors have a lower corporate share (odds ratio ~0.55)
and slightly fewer patents. If analyze.py does not recover a negative, roughly
significant beta_arab here, the pipeline is broken.
"""
import numpy as np
import pandas as pd

import names as nm

rng = np.random.default_rng(7)
N_INV = 4000
ARAB_FRAC = 0.06
TRUE_LOG_OR = np.log(0.55)

ARAB_SUR = ["Haddad", "Khoury", "Zoabi", "Jabareen", "Agbaria", "Mansour",
            "Sakhnin", "Abu Ahmad", "Abdel Rahman", "Tatour", "Zahalka",
            "Masalha", "Khatib", "Younis", "Darwish", "Nassar", "Bishara"]
JEW_SUR = ["Cohen", "Levy", "Mizrahi", "Friedman", "Katz", "Shapira",
           "Rosenberg", "Ben Ami", "Peretz", "Biton", "Golan", "Weiss",
           "Rabinovich", "Goldstein", "Barzilai", "Shalev", "Dayan"]
AMB_SUR = ["Halabi", "Elias", "Amar", "Shami", "Sultan"]

ARAB_CITY = ["Nazareth", "Sakhnin", "Umm al Fahm", "Tamra", "Shefaram",
             "Taibe", "Rahat", "Haifa", "Tira", "Kafr Qasim"]
JEW_CITY = ["Tel Aviv", "Herzliya", "Ramat Gan", "Petah Tikva", "Haifa",
            "Rehovot", "Netanya", "Jerusalem", "Beer Sheva", "Raanana"]

FIELDS = list("GHCABF")
FIELD_P_J = [.34, .30, .12, .08, .09, .07]
FIELD_P_A = [.26, .24, .20, .12, .11, .07]

N_FIRMS = 420
firms = [f"F{i:04d}" for i in range(N_FIRMS)]
firm_size = rng.lognormal(1.6, 1.0, N_FIRMS)

rows = []
pid = 0
for i in range(N_INV):
    r = rng.random()
    if r < 0.04:
        eth, sur = "amb", rng.choice(AMB_SUR)
    elif r < 0.04 + ARAB_FRAC:
        eth, sur = "arab", rng.choice(ARAB_SUR)
    else:
        eth, sur = "jew", rng.choice(JEW_SUR)

    arab = eth == "arab"
    city = rng.choice(ARAB_CITY if arab else JEW_CITY)
    field = rng.choice(FIELDS, p=FIELD_P_A if arab else FIELD_P_J)
    first_year = int(rng.integers(2007, 2021))          # some outside cohort
    n_pat = 1 + rng.poisson(2.2 - (0.35 if arab else 0))

    # planted effect on corporate share
    logit = 2.2 + (TRUE_LOG_OR if arab else 0)
    p_corp = 1 / (1 + np.exp(-logit))

    firm_idx = rng.integers(0, N_FIRMS)
    inv_id = f"I{i:06d}"

    for k in range(n_pat):
        pid += 1
        yr = first_year + int(rng.integers(0, 7))
        if yr > 2024:
            continue
        if rng.random() < p_corp:
            if rng.random() < 0.15:                     # movers
                firm_idx = rng.integers(0, N_FIRMS)
            a_id, a_name, a_type = firms[firm_idx], f"Firm {firm_idx} Ltd", "2"
        else:
            a_id, a_name, a_type = "", "", ""
        if rng.random() < 0.03:                         # university patents
            a_id, a_name, a_type = "U01", "Yissum Research Development Company", "2"
        rows.append({
            "patent_id": f"P{pid:07d}", "filing_year": yr,
            "inventor_id": inv_id, "inventor_last_name": sur,
            "inventor_city": city, "assignee_id": a_id,
            "assignee_name": a_name, "assignee_type": a_type,
            "cpc_section": field,
        })

df = pd.DataFrame(rows)
df.to_csv("raw_records.csv", index=False)
print(f"Synthetic raw_records.csv: {len(df):,} rows, "
      f"{df.inventor_id.nunique():,} inventors")
print(f"Planted true log-OR on corporate share = {TRUE_LOG_OR:.4f} "
      f"(OR {np.exp(TRUE_LOG_OR):.2f})")
