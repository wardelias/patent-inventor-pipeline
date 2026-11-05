"""
START HERE.

This is the only file you need to run. It does everything else for you.

    python START_HERE.py

It will:
  1. install the Python packages you need
  2. ask where you put the downloaded USPTO files
  3. check those files look right
  4. build the dataset and run the analysis

If something goes wrong it will tell you in plain language what to do.
"""
import os
import subprocess
import sys
import traceback

NEEDED = ["pandas", "numpy", "scipy", "patsy", "statsmodels", "matplotlib"]

LINE = "=" * 66


def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(LINE)
    say(f"STEP {n}  {title}")
    say(LINE)


def install_packages():
    step(1, "Checking Python packages")
    missing = []
    for pkg in NEEDED:
        try:
            __import__(pkg)
            say(f"  ok       {pkg}")
        except ImportError:
            missing.append(pkg)
            say(f"  missing  {pkg}")

    if not missing:
        say("\nAll packages present.")
        return

    say(f"\nInstalling {len(missing)} package(s). This may take a few minutes.")
    for flags in (["--user"], ["--break-system-packages"], []):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", *flags, *missing]
            )
            say("Install finished.")
            return
        except subprocess.CalledProcessError:
            continue
    say("\nCould not install automatically. Run this yourself, then start again:")
    say(f"  {sys.executable} -m pip install {' '.join(missing)}")
    sys.exit(1)


def clean_path(raw):
    """Handle quotes, trailing spaces, and Windows/macOS drag-and-drop escapes."""
    p = raw.strip()
    if p and p[0] in "\"'" and p[-1] == p[0]:
        p = p[1:-1]
    p = p.replace("\\ ", " ").strip()
    return os.path.expanduser(p)


def ask_folder():
    step(2, "Where are the downloaded USPTO files?")
    say("Type the folder path and press Enter.")
    say()
    say("  Easiest way: drag the FOLDER from Finder/File Explorer into this")
    say("  window and it will fill in the path for you. Then press Enter.")
    say()
    while True:
        folder = clean_path(input("Folder: "))
        if not folder:
            say("  (nothing typed -- try again)")
            continue
        if not os.path.isdir(folder):
            say(f"  Can't find a folder there: {folder}")
            say("  Check for typos, or drag the folder in instead of typing.")
            continue
        files = [f for f in os.listdir(folder)
                 if f.endswith((".tsv", ".zip", ".csv"))]
        if not files:
            say(f"  That folder has no .tsv or .zip files in it.")
            say(f"  Contents: {os.listdir(folder)[:10]}")
            continue
        say(f"\n  Found {len(files)} data file(s). Good.")
        return folder


def check_files(folder):
    step(3, "Checking the files look right")
    import bulk_load
    found, allfiles = bulk_load.find_files(folder)
    roles = ["location", "inventor", "application", "assignee", "cpc"]
    for role in roles:
        f = found.get(role)
        say(f"  {role:<12} {os.path.basename(f) if f else '*** NOT FOUND ***'}")

    missing = [r for r in roles if r not in found]
    if missing:
        say()
        real = {
            "location": "g_location_disambiguated",
            "inventor": "g_inventor_disambiguated",
            "application": "g_application",
            "assignee": "g_assignee_disambiguated",
            "cpc": "g_cpc_current",
        }
        say("Some files are missing. Go back to the pvgpatdis download page")
        say("and get these:")
        for r in missing:
            say(f"  - {real[r]}")
        say()
        say("Files currently in that folder:")
        for f in allfiles:
            say(f"  {os.path.basename(f)}")
        sys.exit(1)
    say("\nAll five files found.")
    return found


def run_pipeline(folder):
    step(4, "Building the dataset (this is the slow part)")
    if os.path.exists("raw_records.csv"):
        import pandas as pd
        n = sum(1 for _ in open("raw_records.csv", encoding="utf-8")) - 1
        say(f"raw_records.csv already exists ({n:,} rows).")
        say("Re-reading the USPTO files takes 10-40 minutes and would produce")
        say("exactly the same file unless you downloaded newer data.")
        say()
        ans = input("Reuse the existing file and skip ahead? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            say("Reusing raw_records.csv. Skipping to Step 5.")
            return _rest()
        say("Rebuilding from scratch.\n")
    else:
        say("Reading several GB of data. Expect 10-40 minutes. You can leave it.\n")
    import bulk_load
    bulk_load.run(folder)
    return _rest()


def _rest():

    step(5, "Turning patents into one row per inventor")
    import build_panel
    build_panel.build()

    step(6, "Running the analysis")
    import analyze
    df = analyze.load()
    analyze.variance_check(df)
    analyze.descriptives(df)
    analyze.ladder(df)
    try:
        analyze.firm_fixed_effects()
    except Exception as e:
        say(f"\n(firm fixed effects skipped: {e})")


def finish():
    say()
    say(LINE)
    say("DONE")
    say(LINE)
    say("Files created in this folder:")
    for f in ["raw_records.csv", "panel.csv", "flow.csv",
              "results_ladder.csv", "fig_arab_share_by_cohort.png",
              "fig_arab_share_by_field.png", "fig_corp_vs_indep.png"]:
        mark = "yes" if os.path.exists(f) else " - "
        say(f"  [{mark}] {f}")
    say()
    say("Copy everything printed above and send it to Claude.")
    say("The most important part is STEP 6, the GO/NO-GO check.")


def main():
    say(LINE)
    say("Intrapreneurship gaps -- data pipeline")
    say(LINE)
    try:
        install_packages()
        folder = ask_folder()
        check_files(folder)
        run_pipeline(folder)
        finish()
    except KeyboardInterrupt:
        say("\n\nStopped. Nothing was broken -- just run it again when ready.")
    except Exception:
        say()
        say(LINE)
        say("SOMETHING WENT WRONG")
        say(LINE)
        say("This is normal and fixable. Copy everything below and send it")
        say("to Claude -- the last few lines say what the actual problem is.")
        say()
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
