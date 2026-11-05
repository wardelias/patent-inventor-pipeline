#!/usr/bin/env python3
"""
Extract the Israeli-inventor subsample from PatentsView bulk TSV files.

Run this locally, in the folder where you downloaded the zips.
Outputs five small CSVs into ./extract/ which you can upload.

Usage:
    pip install pandas
    python extract_israeli_inventors.py --data-dir "C:/path/to/downloads"

Notes
-----
* Files may be .zip or .tsv; both are handled.
* Pass 1 finds inventors whose recorded location is in Israel.
* Pass 2 re-reads the inventor file to pull the FULL patent history of
  those inventors (including patents filed while they lived elsewhere).
  This is what lets us verify the "no patents before 2010" cohort rule.
* Team size is counted in pass 2 as the number of inventor rows per patent.
"""

import argparse
import glob
import os
import sys
from collections import Counter

import pandas as pd

CHUNK = 500_000


# ---------------------------------------------------------------- helpers

def find_file(data_dir, stem):
    """Locate g_<stem>.tsv / .zip / .tsv.zip in data_dir, case-insensitively."""
    patterns = [
        f"{stem}.tsv", f"{stem}.zip", f"{stem}.tsv.zip",
        f"{stem}.TSV", f"{stem}*.zip", f"{stem}*",
    ]
    for p in patterns:
        hits = glob.glob(os.path.join(data_dir, p))
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def reader(path, usecols=None):
    """Chunked TSV reader that tolerates zip wrappers and schema drift."""
    comp = "zip" if path.lower().endswith(".zip") else "infer"
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        na_filter=False,
        chunksize=CHUNK,
        compression=comp,
        usecols=usecols,
        on_bad_lines="warn",
        engine="c",
    )


def headers(path):
    comp = "zip" if path.lower().endswith(".zip") else "infer"
    return list(pd.read_csv(path, sep="\t", nrows=0, compression=comp).columns)


def pick(cols, *candidates, required=True, label=""):
    """Find the first column matching any candidate substring."""
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for lc, orig in low.items():
            if cand.lower() in lc:
                return orig
    if required:
        sys.exit(f"[!] Could not find a column for {label or candidates[0]}.\n"
                 f"    Available columns: {cols}")
    return None


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="extract")
    ap.add_argument("--country", default="IL")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = {}
    for stem in ["g_location_disambiguated", "g_inventor_disambiguated",
                 "g_application", "g_assignee_disambiguated",
                 "g_cpc_current", "g_patent"]:
        files[stem] = find_file(args.data_dir, stem)
        state = files[stem] if files[stem] else "MISSING"
        log(f"  {stem:32s} -> {state}")

    for req in ["g_location_disambiguated", "g_inventor_disambiguated",
                "g_application", "g_assignee_disambiguated", "g_cpc_current"]:
        if not files[req]:
            sys.exit(f"[!] Required file {req} not found in {args.data_dir}")

    # ---------------------------------------------- 1. Israeli location ids
    log("\n[1/6] Reading locations ...")
    lpath = files["g_location_disambiguated"]
    lcols = headers(lpath)
    c_locid = pick(lcols, "location_id", label="location id")
    c_country = pick(lcols, "disambig_country", "country", label="country")
    c_city = pick(lcols, "disambig_city", "city", label="city")

    loc = pd.concat(
        [ch[ch[c_country].str.upper() == args.country.upper()]
         for ch in reader(lpath)],
        ignore_index=True,
    )
    il_locs = set(loc[c_locid])
    log(f"      {len(il_locs):,} {args.country} location ids")
    loc[[c_locid, c_city, c_country]].rename(
        columns={c_locid: "location_id", c_city: "city", c_country: "country"}
    ).to_csv(os.path.join(args.out_dir, "locations_il.csv"), index=False)

    # ---------------------------------------------- 2. pass 1: IL inventors
    log("[2/6] Pass 1 over inventors (finding Israeli inventors) ...")
    ipath = files["g_inventor_disambiguated"]
    icols = headers(ipath)
    c_pat = pick(icols, "patent_id", label="patent id")
    c_inv = pick(icols, "inventor_id", label="inventor id")
    c_first = pick(icols, "disambig_inventor_name_first", "name_first", "first")
    c_last = pick(icols, "disambig_inventor_name_last", "name_last", "last")
    c_iloc = pick(icols, "location_id", label="inventor location id")
    c_seq = pick(icols, "inventor_sequence", "sequence", required=False)
    c_gender = pick(icols, "gender_code", "gender", required=False)

    il_inventors = set()
    for n, ch in enumerate(reader(ipath), 1):
        hit = ch[ch[c_iloc].isin(il_locs)]
        il_inventors.update(hit[c_inv])
        if n % 20 == 0:
            log(f"      chunk {n}, {len(il_inventors):,} inventors so far")
    log(f"      {len(il_inventors):,} Israeli inventors")

    # ---------------------------------------------- 3. pass 2: full history
    log("[3/6] Pass 2 over inventors (full patent history + team size) ...")
    keep_cols = [c for c in [c_pat, c_inv, c_first, c_last, c_iloc,
                             c_seq, c_gender] if c]
    parts, team = [], Counter()
    target_patents = set()

    for n, ch in enumerate(reader(ipath), 1):
        hit = ch[ch[c_inv].isin(il_inventors)]
        if len(hit):
            parts.append(hit[keep_cols])
            target_patents.update(hit[c_pat])
        if n % 20 == 0:
            log(f"      chunk {n}, {len(target_patents):,} patents so far")

    inv = pd.concat(parts, ignore_index=True)
    del parts

    # second sweep purely to count team size on the target patents
    log("      counting team size ...")
    for ch in reader(ipath, usecols=[c_pat]):
        s = ch[c_pat]
        team.update(s[s.isin(target_patents)].tolist())

    rename = {c_pat: "patent_id", c_inv: "inventor_id", c_first: "name_first",
              c_last: "name_last", c_iloc: "location_id"}
    if c_seq:
        rename[c_seq] = "inventor_sequence"
    if c_gender:
        rename[c_gender] = "gender_code"
    inv = inv.rename(columns=rename)
    inv["team_size"] = inv["patent_id"].map(team).fillna(1).astype(int)
    inv["israeli_location"] = inv["location_id"].isin(il_locs)
    inv.to_csv(os.path.join(args.out_dir, "inventors_il.csv"), index=False)
    log(f"      {len(inv):,} inventor-patent rows, "
        f"{len(target_patents):,} distinct patents")

    # ---------------------------------------------- 4. filing dates
    log("[4/6] Application (filing dates) ...")
    apath = files["g_application"]
    acols = headers(apath)
    a_pat = pick(acols, "patent_id")
    a_date = pick(acols, "filing_date", "date")
    out = pd.concat(
        [ch.loc[ch[a_pat].isin(target_patents), [a_pat, a_date]]
         for ch in reader(apath)],
        ignore_index=True,
    ).rename(columns={a_pat: "patent_id", a_date: "filing_date"})
    out.to_csv(os.path.join(args.out_dir, "application.csv"), index=False)
    log(f"      {len(out):,} rows")

    # ---------------------------------------------- 5. assignees
    log("[5/6] Assignees ...")
    spath = files["g_assignee_disambiguated"]
    scols = headers(spath)
    s_pat = pick(scols, "patent_id")
    s_id = pick(scols, "assignee_id")
    s_org = pick(scols, "disambig_assignee_organization", "organization")
    s_type = pick(scols, "assignee_type", "type", required=False)
    s_seq = pick(scols, "assignee_sequence", "sequence", required=False)
    s_loc = pick(scols, "location_id", required=False)
    s_f = pick(scols, "disambig_assignee_individual_name_first",
               "individual_name_first", required=False)
    s_l = pick(scols, "disambig_assignee_individual_name_last",
               "individual_name_last", required=False)
    scols_keep = [c for c in [s_pat, s_id, s_org, s_type, s_seq, s_loc,
                              s_f, s_l] if c]
    out = pd.concat(
        [ch.loc[ch[s_pat].isin(target_patents), scols_keep]
         for ch in reader(spath)],
        ignore_index=True,
    ).rename(columns={s_pat: "patent_id", s_id: "assignee_id",
                      s_org: "assignee_organization"})
    out.to_csv(os.path.join(args.out_dir, "assignees.csv"), index=False)
    log(f"      {len(out):,} rows "
        f"({len(target_patents) - out['patent_id'].nunique():,} patents have "
        f"NO assignee -> these are the independent/unassigned ones)")

    # ---------------------------------------------- 6. cpc (primary only)
    log("[6/6] CPC (primary classification only) ...")
    cpath = files["g_cpc_current"]
    ccols = headers(cpath)
    p_pat = pick(ccols, "patent_id")
    p_seq = pick(ccols, "cpc_sequence", "sequence")
    p_sec = pick(ccols, "cpc_section", "section")
    p_cls = pick(ccols, "cpc_class", "class")
    p_sub = pick(ccols, "cpc_subclass", "subclass")
    keep = [c for c in [p_pat, p_seq, p_sec, p_cls, p_sub] if c]
    out = pd.concat(
        [ch.loc[ch[p_pat].isin(target_patents) & (ch[p_seq] == "0"), keep]
         for ch in reader(cpath)],
        ignore_index=True,
    ).rename(columns={p_pat: "patent_id", p_sec: "cpc_section",
                      p_cls: "cpc_class", p_sub: "cpc_subclass"})
    out.to_csv(os.path.join(args.out_dir, "cpc_primary.csv"), index=False)
    log(f"      {len(out):,} rows")

    # ---------------------------------------------- optional: patent type
    if files["g_patent"]:
        log("[+]   Patent types ...")
        ppath = files["g_patent"]
        pcols = headers(ppath)
        g_pat = pick(pcols, "patent_id")
        g_type = pick(pcols, "patent_type", "type")
        g_date = pick(pcols, "patent_date", "date", required=False)
        kk = [c for c in [g_pat, g_type, g_date] if c]
        out = pd.concat(
            [ch.loc[ch[g_pat].isin(target_patents), kk] for ch in reader(ppath)],
            ignore_index=True,
        ).rename(columns={g_pat: "patent_id", g_type: "patent_type"})
        out.to_csv(os.path.join(args.out_dir, "patent_type.csv"), index=False)
        log(f"      {len(out):,} rows")

    log(f"\nDone. Upload everything in ./{args.out_dir}/")


if __name__ == "__main__":
    main()
