from pathlib import Path
import pandas as pd


def main() -> None:
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    rows = [
        {"experiment": "smoke", "variable": "QP_S", "iia": None, "status": "placeholder"},
        {"experiment": "smoke", "variable": "NegP", "iia": None, "status": "placeholder"},
    ]
    df = pd.DataFrame(rows)
    out = Path("outputs/results/smoke_placeholder.csv")
    df.to_csv(out, index=False)

    print("Smoke script ran successfully.")
    print(df)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
