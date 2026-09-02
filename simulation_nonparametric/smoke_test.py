import argparse

try:
    from .bridge_estimators import BridgeConfig
    from .cross_fitting import cross_fitting_estimate
    from .dgp import ExtendedLinearDGP
except ImportError:
    from bridge_estimators import BridgeConfig
    from cross_fitting import cross_fitting_estimate
    from dgp import ExtendedLinearDGP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--lambda-bridge", type=float, default=1e-3)
    parser.add_argument("--lambda-adv", type=float, default=1e-2)
    parser.add_argument("--lambda-power", type=float, default=1.0)
    parser.add_argument("--adaptive-lambda", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--penalty", default="l2", choices=["rkhs", "l2"])
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--q-clip", type=float, default=10.0)
    parser.add_argument("--h-clip", type=float, default=20.0)
    parser.add_argument("--score-clip", type=float, default=10.0)
    args = parser.parse_args()

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

    datagen = ExtendedLinearDGP(
        udim=1,
        xdim=3,
        zdim=3,
        wdim=3,
        ddim=1,
        mdim=1,
        proxy_strength=1.5,
        proxy_noise=0.25,
        treatment_proxy_strength=1.1,
        outcome_proxy_strength=1.1,
        fixed_weights=True,
        fixed_weight_scale=0.5,
        seed=123,
    )
    data = datagen.sample_dataset(args.n, seed=123)
    estimates, true_psi = cross_fitting_estimate(datagen, data, config=config, n_splits=args.splits)
    print("population_true_psi", true_psi)
    print("sample_true_psi", estimates["sample_true_psi"])
    for key in ["por", "pipw", "phe1", "phe2", "pmr", "pmr_se"]:
        print(key, estimates[key])


if __name__ == "__main__":
    main()
