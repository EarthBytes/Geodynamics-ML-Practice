# Build and save the simulation database

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_dataset import N_SIMULATIONS, RNG_SEED, save_dataset  # noqa: E402


def main() -> None:
    out = ROOT / "data" / "simulations.csv"
    df = save_dataset(out, n=N_SIMULATIONS, seed=RNG_SEED, include_noise=True)
    print(f"Wrote {len(df)} simulations to {out}")
    print(df.describe().round(4))


if __name__ == "__main__":
    main()
