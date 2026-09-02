Review-version proximal mediation code
======================================

This folder is a self-contained rewrite of the original simulation code for
the review-version bridge-learning rule.

Main change
-----------

The old code estimated later bridge functions by plugging in fitted lower-stage
bridges directly. This version keeps the first-stage bridges nearly unchanged,
but estimates later stages with adjacent/all-stage joint minimax systems:

- First stage:
  - `h2` is estimated from the moment for `Y - h2`.
  - `q0` is estimated from the moment for `A q0 - 1`.
- Second stage:
  - `h1` is estimated jointly with an auxiliary `h2_aux`; final inference still
    uses the first-stage `h2`.
  - `q1` is estimated jointly with an auxiliary `q0_aux`; final inference still
    uses the first-stage `q0`.
- Third stage:
  - `h0` is estimated jointly with auxiliary `h1_aux` and `h2_aux`; final
    inference still uses first-stage `h2`, second-stage `h1`, and third-stage
    `h0`.
  - `q2` is estimated jointly with auxiliary `q1_aux` and `q0_aux`; final
    inference still uses first-stage `q0`, second-stage `q1`, and third-stage
    `q2`.

CUDA
----

The estimator code uses PyTorch. It automatically selects CUDA when available:

```bash
python main.py --device cuda
```

If CUDA is unavailable, use `--device cpu` or leave `--device auto`.

Quick smoke test
----------------

On a server with PyTorch installed:

```bash
python smoke_test.py --device cuda
```

Run a small simulation:

```bash
python main.py --device cuda --preset three_groups --sample-sizes 200 --experiments 3
```

Controller arguments
--------------------

`main.py` is the experiment controller. Important arguments:

```bash
bash run_server.sh
```

For the final two-GPU run, use:

```bash
nohup bash run_final_two_gpu.sh > result/final_launcher.log 2>&1 &
```

This starts group 1 on GPU 0, then group 2 on GPU 0 after group 1 finishes.
Group 3 runs at the same time on GPU 1. All three groups use float64, sample
sizes `1000,2000,3000,4000`, 1000 seeds per sample size, and update both CSV
result files and summaries every 10 seeds.

Equivalent explicit command:

```bash
python main.py \
  --device cuda \
  --preset three_groups \
  --sample-sizes 1000,2000,3000,4000 \
  --experiments 1000 \
  --dtype float64 \
  --splits 5 \
  --output-dir result \
  --save-every 10
```

The default `three_groups` preset runs:

- `group133311_linear_fixed050`: dimensions `(1,3,3,3,1,1)`, fixed midpoint weights
  scaled by `0.50`.
- `group133311_proxyquad010_fixed050`: same dimensions and weight scale, with a
  mild centered-square term added to proxy generation.
- `group234421_linear_fixed035`: dimensions `(2,3,4,4,2,1)`, fixed midpoint weights
  scaled by `0.35`.

Default tuning choices:

- `--penalty l2`: matches the review-version objectives that regularize empirical
  squared bridge values.
- `--lambda-bridge 0.001 --lambda-adv 0.01`: fixed regularization that performed
  best in local CUDA checks after standardization.
- `--gamma 0`: uses a separate median-heuristic RBF bandwidth for every bridge and
  adversarial feature set.
- `--q-clip 10 --h-clip 20 --score-clip 10`: finite-sample stabilization. The
  main source of occasional huge MSE was explosive q-bridge predictions, especially
  `q1`, which can make individual PMR scores enormous. Use `0` for any clip value
  to disable that truncation.
- Kernel inputs are standardized within each training fold by default. Use
  `--no-standardize` only for reproducing older behavior.
- The three-group preset uses fixed deterministic DGP weights, so Monte Carlo
  replicates differ by sampled observations rather than by a randomly easier or
  harder DGP.
- Add `--adaptive-lambda --lambda-bridge 0.5 --lambda-adv 0.5` if you want
  sample-size scaling such as `0.5 / n_train`.

Proxy-quality knobs for sensitivity checks:

- `--proxy-strength`: multiplies the `U -> Z` and `U -> W` signal.
- `--proxy-noise`: multiplies the noise variance of `Z` and `W`.
- `--treatment-proxy-strength`: multiplies the `Z -> A` signal.
- `--outcome-proxy-strength`: multiplies the `W -> Y` signal.
- `--proxy-square-strength`: adds a mild centered-square term to `Z` and `W`.
- `--outcome-square-strength`: optional outcome nonlinearity; not used in the
  default preset because local checks showed occasional finite-sample outliers.

The preset uses `--proxy-strength 1.5`, `--proxy-noise 0.25`,
`--treatment-proxy-strength 1.1`, and `--outcome-proxy-strength 1.1`. These
settings make `Z` and `W` richer and cleaner than the hidden confounders,
matching the completeness intuition.

Outputs
-------

Each run writes:

- `run_config.json`: dimensions, sample sizes, hyperparameters, and CUDA runtime.
- `<group>/n{sample_size}_results.csv`: all per-experiment estimates, MSEs, CI fields.
- `<group>/n{sample_size}_errors.csv`: failed runs, if any.
- `all_experiments_combined.csv`: combined result rows across all sample sizes.
- `experiment_summary_with_ci.csv`: grouped MSE, bias, CI coverage, CI width, and SE summary.

`true_psi` is the population target used for default coverage and MSE. The
sample-specific oracle target is saved separately as `sample_true_psi`; its
coverage diagnostic is `pmr_ci_cover_sample`.

Dependencies
------------

Install a CUDA-enabled PyTorch build for the server, plus:

```bash
pip install numpy pandas tqdm
```
