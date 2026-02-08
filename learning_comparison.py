from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_ORDER = ["warmup_kf", "warmup_dyn", "warmup_crit", "train"]


@dataclass(frozen=True)
class LearningRun:
    label: str
    paths: tuple[Path, ...] = ()
    filters: dict[str, object] = field(default_factory=dict)
    train_spec: "TrainingSpec | None" = None

    @classmethod
    def from_paths(
        cls,
        label: str,
        paths: Iterable[str | Path],
        filters: Mapping[str, object] | None = None,
    ) -> "LearningRun":
        return cls(
            label=label,
            paths=tuple(Path(p) for p in paths),
            filters=dict(filters) if filters else {},
        )

    @classmethod
    def from_training(
        cls,
        label: str,
        train_spec: "TrainingSpec",
        filters: Mapping[str, object] | None = None,
    ) -> "LearningRun":
        return cls(
            label=label,
            paths=(),
            filters=dict(filters) if filters else {},
            train_spec=train_spec,
        )


@dataclass(frozen=True)
class TrainingSpec:
    seeds: Sequence[int] | None = None
    seed: int | None = None
    window_size: int | None = None
    lam: float | None = None
    networksize: int | None = None
    learnrate: float | None = None
    eval_every: int | None = None
    cfg_overrides: dict[str, object] = field(default_factory=dict)
    cache_path: str | Path = "returns.parquet"
    data_bundle: object | None = None
    save_best_path: str | Path | None = None
    evaluate_best_on_test: bool = False
    eval_on_validation: bool = False
    verbose: bool = False
    print_results: bool = False
    stats_csv_path: str | Path | None = None
    overwrite_stats: bool = True


class LearningComparison:
    def __init__(
        self,
        runs: Iterable[LearningRun],
        phases: Sequence[str] | None = None,
        use_global_step: bool = False,
        smooth_window: int | None = None,
        prefer_train_phase: bool = True,
    ) -> None:
        self.runs = list(runs)
        self.phases = tuple(phases) if phases is not None else None
        self.use_global_step = bool(use_global_step)
        self.smooth_window = smooth_window
        self.prefer_train_phase = prefer_train_phase

    @classmethod
    def from_paths(
        cls,
        labeled_paths: Mapping[str, Iterable[str | Path]],
        phases: Sequence[str] | None = None,
        use_global_step: bool = False,
        smooth_window: int | None = None,
        prefer_train_phase: bool = True,
    ) -> "LearningComparison":
        runs = [
            LearningRun.from_paths(label, paths) for label, paths in labeled_paths.items()
        ]
        return cls(
            runs=runs,
            phases=phases,
            use_global_step=use_global_step,
            smooth_window=smooth_window,
            prefer_train_phase=prefer_train_phase,
        )

    @classmethod
    def from_training_specs(
        cls,
        specs: Mapping[str, TrainingSpec],
        phases: Sequence[str] | None = None,
        use_global_step: bool = False,
        smooth_window: int | None = None,
        prefer_train_phase: bool = True,
    ) -> "LearningComparison":
        runs = [
            LearningRun.from_training(label, spec) for label, spec in specs.items()
        ]
        return cls(
            runs=runs,
            phases=phases,
            use_global_step=use_global_step,
            smooth_window=smooth_window,
            prefer_train_phase=prefer_train_phase,
        )

    def prepare(self) -> dict[str, pd.DataFrame]:
        prepared: dict[str, pd.DataFrame] = {}
        for run in self.runs:
            if not run.paths:
                continue
            df = self._load_run(run)
            if df.empty:
                continue
            df = self._prepare_dataframe(df)
            if df.empty:
                continue
            prepared[run.label] = df
        return prepared

    def plot(
        self,
        out_path: str | Path | None = None,
        include_std: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 4),
        show: bool = True,
    ) -> Path | None:
        data = self.prepare()
        if not data:
            print("No validation Sharpe data found to plot.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        for label, df in data.items():
            ax.plot(df["step"], df["mean"], label=label)
            if include_std and (df["std"] > 0).any():
                ax.fill_between(
                    df["step"],
                    df["mean"] - df["std"],
                    df["mean"] + df["std"],
                    alpha=0.2,
                )

        ax.set_title(title or "Validation Sharpe Comparison")
        ax.set_xlabel("Global Step" if self.use_global_step else "Epoch")
        ax.set_ylabel("Validation Sharpe")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            return out_path
        if show:
            plt.show()
        return None

    def run_training(self, out_dir: str | Path = "learning_runs") -> dict[str, tuple[Path, ...]]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        updated_runs: list[LearningRun] = []
        outputs: dict[str, tuple[Path, ...]] = {}

        for run in self.runs:
            spec = run.train_spec
            if spec is None:
                updated_runs.append(run)
                continue

            stats_paths = self._run_training_spec(run.label, spec, out_dir)
            updated_runs.append(
                LearningRun(
                    label=run.label,
                    paths=stats_paths,
                    filters=run.filters,
                    train_spec=run.train_spec,
                )
            )
            outputs[run.label] = stats_paths

        self.runs = updated_runs
        return outputs

    def run_training_and_plot(
        self,
        out_dir: str | Path = "learning_runs",
        out_path: str | Path | None = None,
        include_std: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 4),
        show: bool = True,
    ) -> Path | None:
        self.run_training(out_dir=out_dir)
        return self.plot(
            out_path=out_path,
            include_std=include_std,
            title=title,
            figsize=figsize,
            show=show,
        )

    def to_csvs(self, out_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        for label, df in self.prepare().items():
            safe_label = self._sanitize_label(label)
            path = out_dir / f"{safe_label}_val_sharpe_curve.csv"
            df.to_csv(path, index=False)
            outputs[label] = path
        return outputs

    def _run_training_spec(
        self, label: str, spec: TrainingSpec, out_dir: Path
    ) -> tuple[Path, ...]:
        from portfolio_rl.backtest_seed import run_one
        from portfolio_rl.config import TrainConfig

        if spec.eval_on_validation:
            raise ValueError(
                "eval_on_validation=True disables stats CSV output. "
                "Set eval_on_validation=False for learning curve comparison."
            )

        cfg = TrainConfig(
            device="cpu",
            T=15,  # will be reset based on window_size unless overridden
            updates=1000,
            window_size=10,
            print_every=999999,
            use_kf=True,
            episodes_per_batch=1,
            lam=1.0,
            gamma=0.99,
            dyn_enabled=True,
            use_critic=False,
            kappa_unc=0.0,
            dyn_use_sim=True,
            actor_weight=1.0,
            lr_actor=1e-3,
            dyn_sim_M=10,
            dyn_sim_deterministic=True,
            dyn_sim_pl_weight=0.1,
        )
        for key, value in spec.cfg_overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown TrainConfig field: {key}")
            setattr(cfg, key, value)

        if spec.eval_every is not None:
            cfg.print_every = int(spec.eval_every)

        if spec.window_size is not None:
            cfg.window_size = int(spec.window_size)
        if spec.lam is not None:
            cfg.lam = float(spec.lam)
        if spec.learnrate is not None:
            cfg.lr_actor = float(spec.learnrate)

        if "T" not in spec.cfg_overrides:
            cfg.T = cfg.window_size + 5

        window_size = (
            int(spec.window_size) if spec.window_size is not None else int(cfg.window_size)
        )
        lam = float(spec.lam) if spec.lam is not None else float(cfg.lam)
        learnrate = (
            float(spec.learnrate) if spec.learnrate is not None else float(cfg.lr_actor)
        )
        networksize = int(spec.networksize) if spec.networksize is not None else 128

        if spec.seeds:
            seeds = [int(s) for s in spec.seeds]
        else:
            seeds = [int(spec.seed) if spec.seed is not None else 0]
        stats_paths: list[Path] = []
        for seed in seeds:
            stats_path = self._resolve_stats_path(label, out_dir, spec, seed, len(seeds))
            if spec.overwrite_stats and stats_path.exists():
                stats_path.unlink()

            run_one(
                seed=seed,
                window_size=window_size,
                lam=lam,
                cfg=cfg,
                verbose=spec.verbose,
                save_best_path=spec.save_best_path,
                evaluate_best_on_test=spec.evaluate_best_on_test,
                eval_on_validation=spec.eval_on_validation,
                stats_csv_path=str(stats_path),
                networksize=networksize,
                learnrate=learnrate,
                data_bundle=spec.data_bundle,
                print_results=spec.print_results,
                cache_path=str(spec.cache_path),
            )
            stats_paths.append(stats_path)

        return tuple(stats_paths)

    def _resolve_stats_path(
        self,
        label: str,
        out_dir: Path,
        spec: TrainingSpec,
        seed: int,
        seed_count: int,
    ) -> Path:
        if spec.stats_csv_path is not None:
            base = Path(spec.stats_csv_path)
            if seed_count > 1:
                return self._seeded_path(base, seed)
            return base
        safe_label = self._sanitize_label(label)
        return out_dir / f"{safe_label}_seed{seed}_learning_stats.csv"

    def _load_run(self, run: LearningRun) -> pd.DataFrame:
        frames = []
        for path in run.paths:
            if not path.exists():
                raise FileNotFoundError(f"CSV not found: {path}")
            frames.append(pd.read_csv(path))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return self._apply_filters(df, run.filters)

    def _apply_filters(self, df: pd.DataFrame, filters: Mapping[str, object]) -> pd.DataFrame:
        filtered = df
        for col, value in filters.items():
            if col not in filtered.columns:
                raise ValueError(f"Filter column '{col}' not in CSV.")
            series = filtered[col]
            if isinstance(value, (list, tuple, set)):
                filtered = filtered[series.isin(value)]
                continue
            if isinstance(value, float) and pd.api.types.is_numeric_dtype(series):
                series_num = pd.to_numeric(series, errors="coerce")
                filtered = filtered[np.isclose(series_num, float(value))]
            else:
                filtered = filtered[series == value]
        return filtered

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "val_sharpe" not in df.columns:
            raise ValueError("CSV must contain a 'val_sharpe' column.")

        df = df.copy()
        df["val_sharpe"] = pd.to_numeric(df["val_sharpe"], errors="coerce")
        df = df[df["val_sharpe"].notna()]
        if df.empty:
            return df

        if "phase" in df.columns:
            if self.phases is not None:
                df = df[df["phase"].isin(self.phases)]
            elif self.prefer_train_phase and "train" in set(df["phase"].unique()):
                df = df[df["phase"] == "train"]
        if df.empty:
            return df

        df = self._add_step_column(df)

        if "seed" in df.columns:
            seed_step = (
                df.groupby(["seed", "step"], as_index=False)["val_sharpe"]
                .mean()
            )
            grouped = seed_step.groupby("step")["val_sharpe"]
        else:
            grouped = df.groupby("step")["val_sharpe"]

        agg = grouped.agg(["mean", "std", "count"]).reset_index()
        agg = agg.sort_values("step")
        agg["std"] = agg["std"].fillna(0.0)

        if self.smooth_window is not None and self.smooth_window > 1:
            agg["mean"] = agg["mean"].rolling(self.smooth_window, min_periods=1).mean()

        return agg

    def _add_step_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_global_step or "phase" not in df.columns:
            df = df.copy()
            df["step"] = pd.to_numeric(df["epoch"], errors="coerce")
            return df

        phase_offsets = self._phase_offsets(df)
        if not phase_offsets:
            df = df.copy()
            df["step"] = pd.to_numeric(df["epoch"], errors="coerce")
            return df

        df = df.copy()
        df["step"] = pd.to_numeric(df["epoch"], errors="coerce") + df["phase"].map(
            phase_offsets
        )
        return df

    @staticmethod
    def _phase_offsets(df: pd.DataFrame) -> dict[str, int]:
        phase_offsets: dict[str, int] = {}
        offset = 0
        phases_seen = [p for p in PHASE_ORDER if p in set(df["phase"].unique())]
        for phase in df["phase"].unique():
            if phase not in phases_seen:
                phases_seen.append(phase)
        for phase in phases_seen:
            phase_max = df.loc[df["phase"] == phase, "epoch"].max()
            if pd.isna(phase_max):
                continue
            phase_offsets[phase] = offset
            offset += int(phase_max) + 1
        return phase_offsets

    @staticmethod
    def _sanitize_label(label: str) -> str:
        sanitized = label.strip().replace(" ", "_")
        return sanitized.replace("/", "_")

    @staticmethod
    def _seeded_path(path: Path, seed: int) -> Path:
        name = f"{path.stem}_seed{seed}{path.suffix}"
        return path.with_name(name)


def _parse_run_arg(raw: str) -> LearningRun:
    if "=" not in raw:
        raise ValueError("Run must be formatted as label=path[,path2,...]")
    label, path_str = raw.split("=", 1)
    paths = [p for p in path_str.split(",") if p]
    return LearningRun.from_paths(label, paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare validation Sharpe learning curves across settings."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification as label=path[,path2,...].",
    )
    parser.add_argument("--out-path", default=None, help="Path to save the PNG.")
    parser.add_argument(
        "--phases",
        nargs="*",
        default=None,
        help="Phases to include (defaults to train if present).",
    )
    parser.add_argument(
        "--global-step",
        action="store_true",
        help="Use phase-aware global step on the x-axis.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=None,
        help="Rolling window for smoothing the mean curve.",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable std shading.",
    )
    parser.add_argument("--title", default=None, help="Custom plot title.")
    args = parser.parse_args()

    runs = [_parse_run_arg(arg) for arg in args.run]
    comparison = LearningComparison(
        runs=runs,
        phases=args.phases if args.phases else None,
        use_global_step=args.global_step,
        smooth_window=args.smooth,
    )
    comparison.plot(
        out_path=args.out_path,
        include_std=not args.no_std,
        title=args.title,
    )


if __name__ == "__main__":
    main()
