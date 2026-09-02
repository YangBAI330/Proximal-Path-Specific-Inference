import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .bridge_estimators import BridgeConfig
    from .cross_fitting import cross_fitting_estimate
    from .dgp import ExtendedLinearDGP
except ImportError:
    from bridge_estimators import BridgeConfig
    from cross_fitting import cross_fitting_estimate
    from dgp import ExtendedLinearDGP


BASE_DGP = {
    "var": 0.5,
    "proxy_strength": 1.5,
    "proxy_noise": 0.25,
    "treatment_proxy_strength": 1.1,
    "outcome_proxy_strength": 1.1,
    "fixed_weights": True,
    "proxy_square_strength": 0.0,
    "outcome_square_strength": 0.0,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


THREE_GROUP_PRESET = [
    {
        "name": "group133311_linear_fixed050",
        "dims": (1, 3, 3, 3, 1, 1),
        "dgp": {
            **BASE_DGP,
            "fixed_weight_scale": 0.5,
            "proxy_square_strength": 0.0,
            "outcome_square_strength": 0.0,
        },
    },
    {
        "name": "group133311_proxyquad010_fixed050",
        "dims": (1, 3, 3, 3, 1, 1),
        "dgp": {
            **BASE_DGP,
            "fixed_weight_scale": 0.5,
            "proxy_square_strength": 0.10,
            "outcome_square_strength": 0.0,
        },
    },
    {
        "name": "group234421_linear_fixed035",
        "dims": (2, 3, 4, 4, 2, 1),
        "dgp": {
            **BASE_DGP,
            "fixed_weight_scale": 0.35,
            "proxy_square_strength": 0.0,
            "outcome_square_strength": 0.0,
        },
    },
]


def parse_int_list(value):
    return [int(x) for x in value.replace(",", " ").split() if x]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--preset", default="three_groups", choices=["three_groups", "single"])
    parser.add_argument("--group", default="group133311_goodproxy")
    parser.add_argument("--dims", default="1,3,3,3,1,1", help="udim,xdim,zdim,wdim,ddim,mdim")
    parser.add_argument("--sample-sizes", default="1000,2000,3000,4000")
    parser.add_argument("--experiments", type=int, default=1000)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--lambda-bridge", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1e-2)
    parser.add_argument("--lambda-power", type=float, default=1.0, help="with --adaptive-lambda, effective lambda uses lambda / n_train ** lambda_power")
    parser.add_argument("--adaptive-lambda", action="store_true", help="scale lambda values by n_train")
    parser.add_argument("--gamma", type=float, default=0.0, help="RBF gamma; use 0 for per-feature median heuristic")
    parser.add_argument("--penalty", default="l2", choices=["rkhs", "l2"])
    parser.add_argument("--no-standardize", action="store_true", help="disable fold-wise feature standardization before kernels")
    parser.add_argument("--q-clip", type=float, default=10.0, help="clip q bridge predictions to +/- this value; use 0 to disable")
    parser.add_argument("--h-clip", type=float, default=20.0, help="clip h bridge predictions to +/- this value; use 0 to disable")
    parser.add_argument("--score-clip", type=float, default=10.0, help="clip final per-observation PMR scores; use 0 to disable")
    parser.add_argument("--var", type=float, default=0.5)
    parser.add_argument("--proxy-strength", type=float, default=1.5)
    parser.add_argument("--proxy-noise", type=float, default=0.25)
    parser.add_argument("--treatment-proxy-strength", type=float, default=1.1)
    parser.add_argument("--outcome-proxy-strength", type=float, default=1.1)
    parser.add_argument("--fixed-weights", action="store_true", help="use deterministic midpoint DGP weights instead of drawing new weights per replicate")
    parser.add_argument("--fixed-weight-scale", type=float, default=0.6)
    parser.add_argument("--proxy-square-strength", type=float, default=0.0)
    parser.add_argument("--outcome-square-strength", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=52000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def single_group_from_args(args):
    dims = tuple(parse_int_list(args.dims))
    if len(dims) != 6:
        raise ValueError("--dims must contain six integers: udim,xdim,zdim,wdim,ddim,mdim")
    return {
        "name": args.group,
        "dims": dims,
        "dgp": {
            "var": args.var,
            "proxy_strength": args.proxy_strength,
            "proxy_noise": args.proxy_noise,
            "treatment_proxy_strength": args.treatment_proxy_strength,
            "outcome_proxy_strength": args.outcome_proxy_strength,
            "fixed_weights": args.fixed_weights,
            "fixed_weight_scale": args.fixed_weight_scale,
            "proxy_square_strength": args.proxy_square_strength,
            "outcome_square_strength": args.outcome_square_strength,
        },
    }


def experiment_groups(args):
    if args.preset == "three_groups":
        return THREE_GROUP_PRESET
    for group in THREE_GROUP_PRESET:
        if args.group == group["name"]:
            return [group]
    return [single_group_from_args(args)]


def resolve_output_dir(output_dir):
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(SCRIPT_DIR, output_dir)


def configure_torch_runtime(device, dtype):
    try:
        import torch

        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if dtype == "float32":
                torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def runtime_info(device):
    info = {"requested_device": device}
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def write_run_config(args, groups, sample_sizes, output_dir):
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preset": args.preset,
        "groups": [
            {
                "name": group["name"],
                "dimensions": {
                    "udim": group["dims"][0],
                    "xdim": group["dims"][1],
                    "zdim": group["dims"][2],
                    "wdim": group["dims"][3],
                    "ddim": group["dims"][4],
                    "mdim": group["dims"][5],
                },
                "dgp": group["dgp"],
            }
            for group in groups
        ],
        "sample_sizes": sample_sizes,
        "experiments": args.experiments,
        "splits": args.splits,
        "base_seed": args.base_seed,
        "save_every": args.save_every,
        "bridge_config": {
            "lambda_bridge": args.lambda_bridge,
            "lambda_adv": args.lambda_adv,
            "lambda_power": args.lambda_power,
            "adaptive_lambda": args.adaptive_lambda,
            "gamma": args.gamma,
            "penalty": args.penalty,
            "standardize": not args.no_standardize,
            "dtype": args.dtype,
            "device": args.device,
            "q_clip": args.q_clip,
            "h_clip": args.h_clip,
            "score_clip": args.score_clip,
        },
        "runtime": runtime_info(args.device),
    }
    if args.preset == "single" and len(groups) == 1:
        config_name = f"{groups[0]['name']}_run_config.json"
    else:
        config_name = "run_config.json"
    config_file = os.path.join(output_dir, config_name)
    with open(config_file, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return config_file


def add_distribution(row, prefix, values):
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        row[f"{prefix}_mean"] = np.nan
        row[f"{prefix}_std"] = np.nan
        row[f"{prefix}_median"] = np.nan
        row[f"{prefix}_p95"] = np.nan
        row[f"{prefix}_p99"] = np.nan
        row[f"{prefix}_max"] = np.nan
        return
    row[f"{prefix}_mean"] = float(clean.mean())
    row[f"{prefix}_std"] = float(clean.std())
    row[f"{prefix}_median"] = float(clean.median())
    row[f"{prefix}_p95"] = float(clean.quantile(0.95))
    row[f"{prefix}_p99"] = float(clean.quantile(0.99))
    row[f"{prefix}_max"] = float(clean.max())


def collect_error_counts(output_dir):
    counts = {}
    for root, _, files in os.walk(output_dir):
        for name in files:
            if not name.endswith("_errors.csv"):
                continue
            path = os.path.join(root, name)
            try:
                errors = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if errors.empty or "group" not in errors.columns or "sample_size" not in errors.columns:
                continue
            for (group, sample_size), err_df in errors.groupby(["group", "sample_size"]):
                key = (group, int(sample_size))
                counts[key] = counts.get(key, 0) + int(len(err_df))
    return counts


def create_summary_report(output_dir):
    result_files = []
    for root, _, files in os.walk(output_dir):
        for name in files:
            if name.endswith("_results.csv"):
                result_files.append(os.path.join(root, name))

    if not result_files:
        print("No result files found for summary.")
        return None, None

    frames = [pd.read_csv(path) for path in sorted(result_files)]
    combined = pd.concat(frames, ignore_index=True)
    if "population_true_psi" not in combined.columns:
        combined["population_true_psi"] = combined["true_psi"]
    if "sample_true_psi" not in combined.columns:
        combined["sample_true_psi"] = combined["true_psi"]
    combined_file = os.path.join(output_dir, "all_experiments_combined.csv")
    combined.to_csv(combined_file, index=False)

    error_counts = collect_error_counts(output_dir)
    summary_rows = []
    for (group, sample_size), group_df in combined.groupby(["group", "sample_size"]):
        n_errors = error_counts.get((group, int(sample_size)), 0)
        row = {
            "group": group,
            "sample_size": int(sample_size),
            "n_experiments": int(len(group_df)),
            "n_success": int(len(group_df)),
            "n_errors": int(n_errors),
            "n_attempted": int(len(group_df) + n_errors),
            "true_mean": float(group_df["true_psi"].mean()),
            "true_std": float(group_df["true_psi"].std()),
            "population_true_mean": float(group_df["population_true_psi"].mean()),
            "population_true_std": float(group_df["population_true_psi"].std()),
            "sample_true_mean": float(group_df["sample_true_psi"].mean()),
            "sample_true_std": float(group_df["sample_true_psi"].std()),
        }
        if "pmr_ci_cover" in group_df.columns:
            row["pmr_ci_coverage"] = float(group_df["pmr_ci_cover"].mean() * 100.0)
        if "pmr_ci_cover_sample" in group_df.columns:
            row["pmr_ci_coverage_sample_truth"] = float(group_df["pmr_ci_cover_sample"].mean() * 100.0)
        if "pmr_ci_width" in group_df.columns:
            add_distribution(row, "pmr_ci_width", group_df["pmr_ci_width"])
        if "pmr_se" in group_df.columns:
            add_distribution(row, "pmr_se", group_df["pmr_se"])

        for name in ["por", "pipw", "phe1", "phe2", "pmr"]:
            est_col = f"{name}_estimate"
            mse_col = f"{name}_mse"
            if est_col in group_df.columns:
                row[f"{name}_bias_mean"] = float((group_df[est_col] - group_df["population_true_psi"]).mean())
                row[f"{name}_estimate_mean"] = float(group_df[est_col].mean())
            if mse_col in group_df.columns:
                add_distribution(row, f"{name}_mse", group_df[mse_col])
            sample_mse_col = f"{name}_sample_mse"
            if sample_mse_col in group_df.columns:
                add_distribution(row, sample_mse_col, group_df[sample_mse_col])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["group", "sample_size"]).reset_index(drop=True)
    summary_file = os.path.join(output_dir, "experiment_summary_with_ci.csv")
    summary.to_csv(summary_file, index=False)
    return combined_file, summary_file


def main():
    args = parse_args()
    args.output_dir = resolve_output_dir(args.output_dir)
    configure_torch_runtime(args.device, args.dtype)
    groups = experiment_groups(args)
    sample_sizes = parse_int_list(args.sample_sizes)
    os.makedirs(args.output_dir, exist_ok=True)

    config = BridgeConfig(
        lambda_bridge=args.lambda_bridge,
        lambda_adv=args.lambda_adv,
        gamma_h=args.gamma,
        gamma_q=args.gamma,
        gamma_f=args.gamma,
        device=args.device,
        dtype=args.dtype,
        penalty=args.penalty,
        lambda_power=args.lambda_power,
        adaptive_lambda=args.adaptive_lambda,
        standardize=not args.no_standardize,
        q_clip=args.q_clip,
        h_clip=args.h_clip,
        score_clip=args.score_clip,
    )

    config_file = write_run_config(args, groups, sample_sizes, args.output_dir)

    print("=" * 80)
    print("Review bridge simulation")
    print("output_dir:", args.output_dir)
    print("config_file:", config_file)
    print("preset:", args.preset)
    print("groups:", [group["name"] for group in groups])
    print("sample_sizes:", sample_sizes)
    print("experiments:", args.experiments)
    print("device:", args.device, "dtype:", args.dtype)
    print("runtime:", runtime_info(args.device))
    print("total_jobs:", len(groups) * len(sample_sizes) * args.experiments)
    print("start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    for group_idx, group in enumerate(groups):
        group_name = group["name"]
        udim, xdim, zdim, wdim, ddim, mdim = group["dims"]
        dgp_kwargs = dict(group["dgp"])
        group_dir = os.path.join(args.output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        print("-" * 80)
        print("group:", group_name)
        print("dims:", group["dims"])
        print("dgp:", dgp_kwargs)

        for sample_size in sample_sizes:
            rows = []
            csv_filename = os.path.join(group_dir, f"n{sample_size}_results.csv")
            err_filename = os.path.join(group_dir, f"n{sample_size}_errors.csv")
            desc = f"{group_name} n={sample_size}"
            for exp_idx in tqdm(range(args.experiments), desc=desc, ascii=True):
                seed = args.base_seed + group_idx * 1_000_000 + sample_size * 100 + exp_idx
                try:
                    datagen = ExtendedLinearDGP(
                        udim=udim,
                        xdim=xdim,
                        zdim=zdim,
                        wdim=wdim,
                        ddim=ddim,
                        mdim=mdim,
                        seed=seed,
                        **dgp_kwargs,
                    )
                    data_all = datagen.sample_dataset(sample_size, seed=seed)
                    estimates, true_psi = cross_fitting_estimate(
                        datagen=datagen,
                        data_all=data_all,
                        config=config,
                        n_splits=args.splits,
                        seed=42,
                    )
                    population_true_psi = float(estimates["population_true_psi"])
                    sample_true_psi = float(estimates["sample_true_psi"])
                    record = {
                        "group": group_name,
                        "udim": udim,
                        "xdim": xdim,
                        "zdim": zdim,
                        "wdim": wdim,
                        "ddim": ddim,
                        "mdim": mdim,
                        **dgp_kwargs,
                        "sample_size": sample_size,
                        "seed": seed,
                        "true_psi": float(true_psi),
                        "population_true_psi": population_true_psi,
                        "sample_true_psi": sample_true_psi,
                        "por_estimate": estimates["por"],
                        "pipw_estimate": estimates["pipw"],
                        "phe1_estimate": estimates["phe1"],
                        "phe2_estimate": estimates["phe2"],
                        "pmr_estimate": estimates["pmr"],
                        "pmr_se": estimates["pmr_se"],
                        "pmr_ci_lower": estimates["pmr_ci_lower"],
                        "pmr_ci_upper": estimates["pmr_ci_upper"],
                        "pmr_ci_cover": estimates["pmr_ci_cover"],
                        "pmr_ci_cover_sample": estimates["pmr_ci_cover_sample"],
                        "pmr_ci_width": estimates["pmr_ci_width"],
                        "por_mse": float((estimates["por"] - population_true_psi) ** 2),
                        "pipw_mse": float((estimates["pipw"] - population_true_psi) ** 2),
                        "phe1_mse": float((estimates["phe1"] - population_true_psi) ** 2),
                        "phe2_mse": float((estimates["phe2"] - population_true_psi) ** 2),
                        "pmr_mse": float((estimates["pmr"] - population_true_psi) ** 2),
                        "por_sample_mse": float((estimates["por"] - sample_true_psi) ** 2),
                        "pipw_sample_mse": float((estimates["pipw"] - sample_true_psi) ** 2),
                        "phe1_sample_mse": float((estimates["phe1"] - sample_true_psi) ** 2),
                        "phe2_sample_mse": float((estimates["phe2"] - sample_true_psi) ** 2),
                        "pmr_sample_mse": float((estimates["pmr"] - sample_true_psi) ** 2),
                        "experiment_index": exp_idx,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    rows.append(record)
                    if (exp_idx + 1) % args.save_every == 0:
                        pd.DataFrame(rows).to_csv(csv_filename, index=False)
                        summary_root = group_dir if args.preset == "single" else args.output_dir
                        _, summary_file = create_summary_report(summary_root)
                        print(
                            "checkpoint",
                            "group",
                            group_name,
                            "sample_size",
                            sample_size,
                            "seeds",
                            exp_idx + 1,
                            "summary_file",
                            summary_file,
                        )
                except Exception as exc:
                    err = pd.DataFrame(
                        [
                            {
                                "group": group_name,
                                "sample_size": sample_size,
                                "seed": seed,
                                "experiment_index": exp_idx,
                                "error": str(exc),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        ]
                    )
                    err.to_csv(err_filename, mode="a", header=not os.path.exists(err_filename), index=False)
                    if args.fail_fast:
                        raise

            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_filename, index=False)
                summary_root = group_dir if args.preset == "single" else args.output_dir
                _, summary_file = create_summary_report(summary_root)
                print("group", group_name, "sample_size", sample_size, "success", len(rows))
                print("pmr_mse_mean", float(df["pmr_mse"].mean()))
                print("pmr_mse_median", float(df["pmr_mse"].median()))
                print("pmr_ci_cover_population", float(df["pmr_ci_cover"].mean()))
                if "pmr_ci_cover_sample" in df.columns:
                    print("pmr_ci_cover_sample", float(df["pmr_ci_cover_sample"].mean()))
                print("summary_file", summary_file)

    final_summary_root = (
        os.path.join(args.output_dir, groups[0]["name"])
        if args.preset == "single" and len(groups) == 1
        else args.output_dir
    )
    combined_file, summary_file = create_summary_report(final_summary_root)
    print("=" * 80)
    print("finished:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("combined_file:", combined_file)
    print("summary_file:", summary_file)
    if summary_file is not None:
        print(pd.read_csv(summary_file).to_string(index=False))


if __name__ == "__main__":
    main()
