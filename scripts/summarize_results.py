from pathlib import Path
import pandas as pd

RESULT_DIRS = [Path("outputs/results"), Path("results")]


def main() -> None:
    csvs = []
    for d in RESULT_DIRS:
        if d.exists():
            csvs.extend(sorted(d.glob("*.csv")))

    if not csvs:
        print("No CSV result files found.")
        return

    for path in csvs:
        print(f"\n=== {path} ===")
        df = pd.read_csv(path)
        print(df.head())
        print(f"shape = {df.shape}")


if __name__ == "__main__":
    main()
