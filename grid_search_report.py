# portfolio_rl/grid_search_report.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class GridSearchReport:
    lines_path: Path
    raw: pd.DataFrame
    seed_avg: pd.DataFrame
    lambda_avg: pd.DataFrame
    window_avg: pd.DataFrame
    networksize_avg: pd.DataFrame
    learnrate_avg: pd.DataFrame

    @classmethod
    def from_lines_file(cls, lines_path: str | Path) -> "GridSearchReport":
        path = Path(lines_path)
        raw = cls._read_lines(path)
        seed_avg = cls._avg_over_seeds(raw)
        lambda_avg = cls._avg_over_lambda(seed_avg)
        window_avg = cls._avg_over_window(seed_avg)
        networksize_avg = cls._avg_over_networksize(seed_avg)
        learnrate_avg = cls._avg_over_learnrate(seed_avg)
        return cls(
            path,
            raw,
            seed_avg,
            lambda_avg,
            window_avg,
            networksize_avg,
            learnrate_avg,
        )

    def to_excel(self, out_path: str | Path = "grid_search_report.xlsx") -> Path:
        out_path = Path(out_path)
        try:
            with pd.ExcelWriter(out_path) as writer:
                self.raw.to_excel(writer, sheet_name="raw", index=False)
                self.seed_avg.to_excel(writer, sheet_name="seed_avg", index=False)
                self.lambda_avg.to_excel(writer, sheet_name="lambda_avg", index=False)
                self.window_avg.to_excel(writer, sheet_name="window_avg", index=False)
                self.networksize_avg.to_excel(
                    writer, sheet_name="networksize_avg", index=False
                )
                self.learnrate_avg.to_excel(
                    writer, sheet_name="learnrate_avg", index=False
                )
        except ImportError as exc:
            raise ImportError(
                "Writing .xlsx requires openpyxl. Install it or use to_csvs()."
            ) from exc
        return out_path

    def to_csvs(self, out_dir: str | Path = ".") -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "raw": out_dir / "grid_search_raw.csv",
            "seed_avg": out_dir / "grid_search_seed_avg.csv",
            "lambda_avg": out_dir / "grid_search_lambda_avg.csv",
            "window_avg": out_dir / "grid_search_window_avg.csv",
            "networksize_avg": out_dir / "grid_search_networksize_avg.csv",
            "learnrate_avg": out_dir / "grid_search_learnrate_avg.csv",
        }
        self.raw.to_csv(paths["raw"], index=False)
        self.seed_avg.to_csv(paths["seed_avg"], index=False)
        self.lambda_avg.to_csv(paths["lambda_avg"], index=False)
        self.window_avg.to_csv(paths["window_avg"], index=False)
        self.networksize_avg.to_csv(paths["networksize_avg"], index=False)
        self.learnrate_avg.to_csv(paths["learnrate_avg"], index=False)
        return paths

    @staticmethod
    def _read_lines(path: Path) -> pd.DataFrame:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(GridSearchReport._parse_line(line))
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_line(line: str) -> dict[str, float | int]:
        parts = line.split()
        data: dict[str, float | int] = {}
        for part in parts:
            if part == "done":
                continue
            key, value = part.split("=", 1)
            data[key] = value
        learnrate_key = "learnrate" if "learnrate" in data else "leanrate"
        return {
            "seed": int(data["seed"]),
            "window_size": int(data["w"]),
            "lam": float(data["lam"]),
            "networksize": int(data["networksize"]),
            "learnrate": float(data[learnrate_key]),
            "sharpe": float(data["sharpe"]),
            "cumret": float(data["cumret"]),
            "meanret": float(data["meanret"]),
            "stdret": float(data["stdret"]),
        }

    @staticmethod
    def _avg_over_seeds(raw: pd.DataFrame) -> pd.DataFrame:
        grouped = raw.groupby(
            ["window_size", "lam", "networksize", "learnrate"],
            as_index=False,
        ).mean(numeric_only=True)
        return grouped

    @staticmethod
    def _avg_over_lambda(seed_avg: pd.DataFrame) -> pd.DataFrame:
        grouped = seed_avg.groupby("lam", as_index=False).mean(numeric_only=True)
        return grouped

    @staticmethod
    def _avg_over_window(seed_avg: pd.DataFrame) -> pd.DataFrame:
        grouped = seed_avg.groupby("window_size", as_index=False).mean(numeric_only=True)
        return grouped

    @staticmethod
    def _avg_over_networksize(seed_avg: pd.DataFrame) -> pd.DataFrame:
        grouped = seed_avg.groupby("networksize", as_index=False).mean(
            numeric_only=True
        )
        return grouped

    @staticmethod
    def _avg_over_learnrate(seed_avg: pd.DataFrame) -> pd.DataFrame:
        grouped = seed_avg.groupby("learnrate", as_index=False).mean(numeric_only=True)
        return grouped

if __name__ == "__main__":
    report = GridSearchReport.from_lines_file("grid_search_lines.txt")
    #report.to_excel("grid_search_report.xlsx")  # requires openpyxl
    # or:
    report.to_csvs(".")
