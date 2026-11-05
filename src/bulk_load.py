"""
Step 1 (REPLACES extract.py) -- build raw_records.csv from ODP bulk TSVs.

The PatentsView Search API was shut down on 20 March 2026. Bulk download from
the USPTO Open Data Portal is now the only route.

    Dataset: "PatentsView Granted Patent Disambiguated" (pvgpatdis)
    https://data.uspto.gov/bulkdata/datasets/pvgpatdis

Download these five files into one folder (.tsv.zip is fine, no need to unzip):

    g_location_disambiguated      <- small; this is your country filter
    g_inventor_disambiguated      <- large
    g_application                 <- medium
    g_assignee_disambiguated      <- large
    g_cpc_current                 <- very large

Do NOT download the text tables (detail_desc_text, brf_sum_text). Tens of GB,
and you need none of it.

USAGE
    python bulk_load.py --inspect  /path/to/folder
        Prints which file it matched to which role and the real column headers.
        RUN THIS FIRST. Column names may have changed in the ODP migration.

    python bulk_load.py            /path/to/folder
        Streams everything in chunks and writes raw_records.csv.

WHY IT IS FAST ON A LAPTOP
    Israeli patents are a fraction of a percent of USPTO. We resolve Israeli
    location_ids from the small table first, use those to keep a tiny slice of
    the inventor table, then use the resulting patent_id set to filter every
    other table. Peak memory stays low even though the inputs total ~20 GB.
"""
import glob
import os
import sys

import pandas as pd

CHUNK = 500_000

# role -> (filename must contain ALL of these, must contain NONE of these)
FILE_PATTERNS = {
    "location":  (["location", "disambig"], []),
    "inventor":  (["inventor", "disambig"], ["not_disambig"]),
    "application": (["application"], ["cpc", "assignee", "inventor"]),
    "assignee":  (["assignee", "disambig"], ["not_disambig"]),
    "cpc":       (["cpc", "current"], ["at_issue"]),
}

# logical column -> candidate real names, in order of preference
COLS = {
    "location_id":   ["location_id", "id", "rawlocation_id"],
    "country":       ["disambig_country", "country", "location_country"],
    "city":          ["disambig_city", "city", "location_city"],
    "patent_id":     ["patent_id", "id"],
    "inventor_id":   ["inventor_id", "disambig_inventor_id"],
    "last_name":     ["disambig_inventor_name_last", "inventor_name_last",
                      "name_last", "last_name"],
    "filing_date":   ["filing_date", "date", "application_filing_date"],
    "assignee_id":   ["assignee_id", "disambig_assignee_id"],
    "assignee_org":  ["disambig_assignee_organization",
                      "assignee_organization", "organization"],
    "assignee_type": ["assignee_type", "type"],
    "assignee_seq":  ["assignee_sequence", "sequence"],
    "cpc_section":   ["cpc_section", "section_id", "section"],
    "cpc_seq":       ["cpc_sequence", "sequence"],
}


def find_files(folder):
    found = {}
    allfiles = [f for f in glob.glob(os.path.join(folder, "*"))
                if f.endswith((".tsv", ".tsv.zip", ".zip", ".csv", ".csv.zip"))]
    for role, (must, mustnot) in FILE_PATTERNS.items():
        for f in allfiles:
            base = os.path.basename(f).lower()
            if all(m in base for m in must) and not any(x in base for x in mustnot):
                found[role] = f
                break
    return found, allfiles


def header(path):
    return list(pd.read_csv(path, sep="\t", nrows=0, dtype=str).columns)


def pick(cols, logical, required=True, role=""):
    for cand in COLS[logical]:
        if cand in cols:
            return cand
    if required:
        raise KeyError(
            f"[{role}] could not find a column for '{logical}'.\n"
            f"  tried: {COLS[logical]}\n  file has: {cols}\n"
            f"  -> add the real name to COLS['{logical}'] at the top of this file."
        )
    return None


_ROWS = {}


def reader(path, usecols, role=""):
    """
    Stream a table in chunks with the fast C parser.

    Malformed lines are skipped. A callable on_bad_lines would let us count
    them exactly, but pandas only accepts one with the Python engine, which
    is orders of magnitude slower and cannot handle a 20 GB CPC file. Instead
    we report rows parsed per table (see the summary at the end of run()), so
    a catastrophic loss is visible even though individual bad lines are not.
    """
    _ROWS.setdefault(role, 0)
    return pd.read_csv(path, sep="\t", dtype=str, usecols=usecols,
                       chunksize=CHUNK, on_bad_lines="skip",
                       engine="c", low_memory=False)


def inspect(folder):
    found, allfiles = find_files(folder)
    print(f"Scanned {folder} -- {len(allfiles)} candidate files\n")
    for role in FILE_PATTERNS:
        f = found.get(role)
        if not f:
            print(f"  {role:<12} NOT FOUND")
            continue
        print(f"  {role:<12} {os.path.basename(f)}")
        try:
            print(f"    columns: {header(f)}")
        except Exception as e:
            print(f"    !! could not read header: {e}")
    missing = set(FILE_PATTERNS) - set(found)
    if missing:
        print(f"\nMissing roles: {sorted(missing)}")
        print("Files present:", [os.path.basename(f) for f in allfiles])


def run(folder, out="raw_records.csv", country="IL"):
    found, _ = find_files(folder)
    missing = set(FILE_PATTERNS) - set(found)
    if missing:
        sys.exit(f"Missing files for: {sorted(missing)}. Run --inspect first.")

    # ---- 1. Israeli locations (small table, full read) --------------------
    h = header(found["location"])
    c_loc, c_ctry, c_city = (pick(h, "location_id", role="location"),
                             pick(h, "country", role="location"),
                             pick(h, "city", role="location"))
    loc = pd.read_csv(found["location"], sep="\t", dtype=str,
                      usecols=[c_loc, c_ctry, c_city])
    loc = loc[loc[c_ctry].str.upper().str.strip() == country]
    city_of = dict(zip(loc[c_loc], loc[c_city].fillna("")))
    print(f"1. {country} locations: {len(city_of):,}")
    if not city_of:
        sys.exit(f"No locations with country == '{country}'. "
                 f"Check the country coding in {found['location']}.")

    # ---- 2. inventors at those locations ----------------------------------
    h = header(found["inventor"])
    c_pat = pick(h, "patent_id", role="inventor")
    c_inv = pick(h, "inventor_id", role="inventor")
    c_last = pick(h, "last_name", role="inventor")
    c_iloc = pick(h, "location_id", role="inventor")
    parts = []
    for i, ch in enumerate(reader(found["inventor"], [c_pat, c_inv, c_last, c_iloc], "inventor")):
        _ROWS["inventor"] += len(ch)
        ch = ch[ch[c_iloc].isin(city_of)]
        if len(ch):
            parts.append(ch)
        print(f"   inventor chunk {i+1}, kept {sum(len(p) for p in parts):,}",
              end="\r", flush=True)
    inv = pd.concat(parts, ignore_index=True)
    inv["inventor_city"] = inv[c_iloc].map(city_of)
    inv = inv.rename(columns={c_pat: "patent_id", c_inv: "inventor_id",
                              c_last: "inventor_last_name"})
    inv = inv[["patent_id", "inventor_id", "inventor_last_name", "inventor_city"]]
    pats = set(inv["patent_id"])
    print(f"\n2. inventor-patent rows: {len(inv):,}  "
          f"({inv.inventor_id.nunique():,} inventors, {len(pats):,} patents)")

    # ---- 3. filing dates ---------------------------------------------------
    h = header(found["application"])
    c_pat = pick(h, "patent_id", role="application")
    c_date = pick(h, "filing_date", role="application")
    parts = []
    for ch in reader(found["application"], [c_pat, c_date], "application"):
        _ROWS["application"] += len(ch)
        ch = ch[ch[c_pat].isin(pats)]
        if len(ch):
            parts.append(ch)
    app = pd.concat(parts, ignore_index=True).rename(
        columns={c_pat: "patent_id", c_date: "filing_date"})
    app["filing_year"] = pd.to_numeric(
        app["filing_date"].astype(str).str[:4], errors="coerce")
    app = app.dropna(subset=["filing_year"]).drop_duplicates("patent_id")
    app["filing_year"] = app["filing_year"].astype(int)
    print(f"3. filing years: {len(app):,}")

    # ---- 4. assignees (first listed per patent) ----------------------------
    h = header(found["assignee"])
    c_pat = pick(h, "patent_id", role="assignee")
    c_aid = pick(h, "assignee_id", role="assignee")
    c_org = pick(h, "assignee_org", role="assignee")
    c_typ = pick(h, "assignee_type", required=False, role="assignee")
    c_seq = pick(h, "assignee_seq", required=False, role="assignee")
    use = [c for c in [c_pat, c_aid, c_org, c_typ, c_seq] if c]
    parts = []
    for ch in reader(found["assignee"], use, "assignee"):
        _ROWS["assignee"] += len(ch)
        ch = ch[ch[c_pat].isin(pats)]
        if len(ch):
            parts.append(ch)
    asg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
    if c_seq and len(asg):
        asg[c_seq] = pd.to_numeric(asg[c_seq], errors="coerce").fillna(0)
        asg = asg.sort_values(c_seq)
    asg = asg.drop_duplicates(c_pat)
    ren = {c_pat: "patent_id", c_aid: "assignee_id", c_org: "assignee_name"}
    if c_typ:
        ren[c_typ] = "assignee_type"
    asg = asg.rename(columns=ren)
    if "assignee_type" not in asg:
        asg["assignee_type"] = ""
    asg = asg[["patent_id", "assignee_id", "assignee_name", "assignee_type"]]
    print(f"4. assignees: {len(asg):,} patents assigned "
          f"({len(pats) - len(asg):,} unassigned -> independent)")

    # ---- 5. CPC section (primary) ------------------------------------------
    h = header(found["cpc"])
    c_pat = pick(h, "patent_id", role="cpc")
    c_sec = pick(h, "cpc_section", role="cpc")
    c_seq = pick(h, "cpc_seq", required=False, role="cpc")
    use = [c for c in [c_pat, c_sec, c_seq] if c]
    parts = []
    for i, ch in enumerate(reader(found["cpc"], use, "cpc")):
        _ROWS["cpc"] += len(ch)
        ch = ch[ch[c_pat].isin(pats)]
        if len(ch):
            parts.append(ch)
        print(f"   cpc chunk {i+1}", end="\r", flush=True)
    cpc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
    if c_seq and len(cpc):
        cpc[c_seq] = pd.to_numeric(cpc[c_seq], errors="coerce").fillna(0)
        cpc = cpc.sort_values(c_seq)
    cpc = cpc.drop_duplicates(c_pat).rename(
        columns={c_pat: "patent_id", c_sec: "cpc_section"})
    cpc["cpc_section"] = cpc["cpc_section"].astype(str).str[:1]
    cpc = cpc[["patent_id", "cpc_section"]]
    print(f"\n5. cpc sections: {len(cpc):,}")

    # ---- 6. assemble to the raw_records.csv contract ------------------------
    df = (inv.merge(app[["patent_id", "filing_year"]], on="patent_id", how="inner")
             .merge(asg, on="patent_id", how="left")
             .merge(cpc, on="patent_id", how="left"))
    for c in ["assignee_id", "assignee_name", "assignee_type", "cpc_section"]:
        df[c] = df[c].fillna("")
    df = df[["patent_id", "filing_year", "inventor_id", "inventor_last_name",
             "inventor_city", "assignee_id", "assignee_name", "assignee_type",
             "cpc_section"]]
    df.to_csv(out, index=False)
    print("\n   rows parsed per source table (sanity check):")
    for k, v in _ROWS.items():
        print(f"     {k:<14} {v:>12,}")
    print(f"\n6. wrote {out}: {len(df):,} rows, "
          f"{df.inventor_id.nunique():,} inventors, "
          f"filing years {df.filing_year.min()}-{df.filing_year.max()}")
    print("   next:  python build_panel.py")
    return df


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    folder = args[0] if args else "."
    if "--inspect" in sys.argv:
        inspect(folder)
    else:
        run(folder)
