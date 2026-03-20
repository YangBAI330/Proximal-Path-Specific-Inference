import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold
from scipy import stats
import linear_dgp
import allestimator 

estimator = allestimator.AllEstimator

def compute_influence_function_pmr(model, data, psi_hat):
    """
    Calculate the influence function (IF) of the PMR estimator
    Calculate the influence function of each sample based on the latest PMR formula:
    EIF = A*q0*(h1-h0) + (1-A)*q1*(h2-h1) + A*q2*(Y-h2) + h0 - psi_hat
    """
   # Get data
    a = data['a'][:, 0].astype(float)
    y = data['y'][:, 0].astype(float)
    x = data['x']
    w = data['w']
    z = data['z']
    d = data['d']
    m = data['m']
    
    # indicator function
    a1_ind = a
    a0_ind = 1.0 - a
    
    # ----h function prediction ----
    wx_h0 = np.hstack((w, x))
    h0_pred = model.h0_fn(wx_h0).flatten()
    
    wx_h1 = np.hstack((w, d, x))
    h1_pred = model.h1_fn(wx_h1).flatten()
    
    wx_h2 = np.hstack((w, x, m, d))
    h2_pred = model.h2_fn(wx_h2).flatten()
    
    # ----q function prediction ----
    zx_q0 = np.hstack((z, x))
    q0_pred = model.q0_fn(zx_q0).flatten()
    
    zx_q1 = np.hstack((z, d, x))
    q1_pred = model.q1_fn(zx_q1).flatten()
    
    zx_q2 = np.hstack((z, x, m, d))
    q2_pred = model.q2_fn(zx_q2).flatten()
    
    # ----Calculate EIF ----
    term1 = a1_ind * q0_pred * (h1_pred - h0_pred)
    term2 = a0_ind * q1_pred * (h2_pred - h1_pred)
    term3 = a1_ind * q2_pred * (y - h2_pred)
    term4 = h0_pred
    
    eif = term1 + term2 + term3 + term4 - psi_hat
    return eif

def cross_fitting_estimate(datagen, data_all, n_splits=5):
    """Use cross-fitting to estimate potential outcomes while calculating confidence intervals based on influence functions"""
    
    # Calculate the true value
    true_psi = datagen.true_psi(data=data_all)
    
    #Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store the estimated value of each fold
    fold_estimates_por = []
    fold_estimates_pipw = []
    fold_estimates_phe1 = []
    fold_estimates_phe2 = []
    fold_estimates_pmr = []
    
    # Store the IF value of each sample (used to calculate the variance)
    n_total = len(data_all['x'])
    sample_if_pmr = np.zeros(n_total)
    
    indices = np.arange(n_total)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        # Split data
        data_fit = {k: v[train_idx] for k, v in data_all.items()}
        data_test = {k: v[test_idx] for k, v in data_all.items()}
        
        try:
            # Training model
            model = estimator()
            model.fit(fit_data=data_fit)
            
            # Evaluate all estimators on the test set
            por_estimates = np.array(model.evaluate_por(eval_data=data_test))
            pipw_estimates = np.array(model.evaluate_pipw(eval_data=data_test))
            phe1_estimates = np.array(model.evaluate_phe1(eval_data=data_test))
            phe2_estimates = np.array(model.evaluate_phe2(eval_data=data_test))
            pmr_estimates = np.array(model.evaluate_pmr(eval_data=data_test))
            
            # Calculate the pmr point estimate for the test set
            fold_pmr_estimate = np.mean(pmr_estimates)
            
            # Calculate the influence function on the test set
            test_if = compute_influence_function_pmr(model, data_test, fold_pmr_estimate)
            sample_if_pmr[test_idx] = test_if
            
            # Store the estimated value for each fold
            fold_estimates_por.append(np.mean(por_estimates))
            fold_estimates_pipw.append(np.mean(pipw_estimates))
            fold_estimates_phe1.append(np.mean(phe1_estimates))
            fold_estimates_phe2.append(np.mean(phe2_estimates))
            fold_estimates_pmr.append(fold_pmr_estimate)
            
        except Exception as e:
            error_msg = f"The {fold_idx+1}th fold failed: {str(e)}"
            print(f"Error details: {error_msg}")
            raise RuntimeError(error_msg) from e
            
    # Calculate the final estimate of the cross-fit (average of all folds)
    if len(fold_estimates_por) > 0:
        cross_fit_estimate_por = np.mean(fold_estimates_por)
        cross_fit_estimate_pipw = np.mean(fold_estimates_pipw)
        cross_fit_estimate_phe1 = np.mean(fold_estimates_phe1)
        cross_fit_estimate_phe2 = np.mean(fold_estimates_phe2)
        cross_fit_estimate_pmr = np.mean(fold_estimates_pmr)
    else:
        cross_fit_estimate_por = 0.0
        cross_fit_estimate_pipw = 0.0
        cross_fit_estimate_phe1 = 0.0
        cross_fit_estimate_phe2 = 0.0
        cross_fit_estimate_pmr = 0.0

    # Calculating confidence intervals for PMR -influence function based approach
    if len(fold_estimates_pmr) > 1:
        if_variance = np.var(sample_if_pmr, ddof=1)
        pmr_variance_if = if_variance / n_total
        pmr_se_if = np.sqrt(pmr_variance_if)
        
        z_critical = stats.norm.ppf(0.975)  # 95% confidence level
        pmr_ci_lower = cross_fit_estimate_pmr - z_critical * pmr_se_if
        pmr_ci_upper = cross_fit_estimate_pmr + z_critical * pmr_se_if
        pmr_ci_cover = (pmr_ci_lower <= true_psi <= pmr_ci_upper)
        
        # Original method as comparison
        fold_variance = np.var(np.array(fold_estimates_pmr), ddof=1)
        pmr_se_original = np.sqrt(fold_variance / len(fold_estimates_pmr))
        pmr_se = pmr_se_if
    else:
        pmr_se = 0.0
        pmr_se_original = 0.0
        pmr_ci_lower = cross_fit_estimate_pmr
        pmr_ci_upper = cross_fit_estimate_pmr
        pmr_ci_cover = True
        if_variance = 0.0
    
    return {
        'por': cross_fit_estimate_por,
        'pipw': cross_fit_estimate_pipw,
        'phe1': cross_fit_estimate_phe1, 
        'phe2': cross_fit_estimate_phe2, 
        'pmr': cross_fit_estimate_pmr,
        'pmr_se': pmr_se,
        'pmr_se_original': pmr_se_original,
        'pmr_ci_lower': pmr_ci_lower,
        'pmr_ci_upper': pmr_ci_upper,
        'pmr_ci_cover': pmr_ci_cover,
        'pmr_ci_width': pmr_ci_upper - pmr_ci_lower,
        'pmr_if_variance': if_variance
    }, true_psi

def main():
    results_dir = "/home/wsh/simulationby"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    experiment_configs = [
        {
            'name': 'group233321_nested_dims',
            # Dimension order: udim, xdim, zdim, wdim, ddim, mdim
            'dimensions': (2, 3, 3, 3, 2, 1), 
            'sample_sizes': [200, 500, 800, 1000, 2000],
            'n_experiments': 300,
            'n_splits': 5
        },
    ]
    
    total_experiments = sum(len(c['sample_sizes']) * c['n_experiments'] for c in experiment_configs)
    
    print("="*80)
    print(f"Start large-scale experiment (with confidence interval calculation)")
    print(f"Destination folder: {results_dir}")
    print(f"Total number of experiments: {total_experiments}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    for config_idx, config in enumerate(experiment_configs, 1):
        group_name = config['name']
        udim, xdim, zdim, wdim, ddim, mdim = config['dimensions']
        
        group_dir = os.path.join(results_dir, group_name)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
            
        for sample_size in config['sample_sizes']:
            print(f"\n--- sample size n = {sample_size} ---")
            csv_filename = os.path.join(group_dir, f"n{sample_size}_results.csv")
            all_results = []
            base_seed = config_idx * 32000 + sample_size * 100
            
            for exp_idx in tqdm(range(config['n_experiments']), desc=f"n={sample_size}"):
                seed = base_seed + exp_idx
                
                try:
                    # Use the new data generator
                    datagen = linear_dgp.ExtendedLinearDGP(
                        udim=udim, 
                        xdim=xdim, 
                        zdim=zdim, 
                        wdim=wdim, 
                        ddim=ddim, 
                        mdim=mdim, 
                        seed=seed
                    )
                    
                    data_all = datagen.sample_dataset(sample_size, seed=seed)
                    
                    estimates_dict, true_psi = cross_fitting_estimate(
                        datagen=datagen,
                        data_all=data_all,
                        n_splits=config['n_splits']
                    )
                    
                    result_record = {
                        'group': group_name,
                        'udim': udim, 'xdim': xdim, 'zdim': zdim, 'wdim': wdim, 'ddim': ddim, 'mdim': mdim,
                        'sample_size': sample_size,
                        'seed': seed,
                        'true_psi': float(true_psi),
                        'por_estimate': float(estimates_dict['por']),
                        'pipw_estimate': float(estimates_dict['pipw']),
                        'phe1_estimate': float(estimates_dict['phe1']),
                        'phe2_estimate': float(estimates_dict['phe2']),
                        'pmr_estimate': float(estimates_dict['pmr']),
                        'pmr_se': float(estimates_dict['pmr_se']),
                        'pmr_ci_lower': float(estimates_dict['pmr_ci_lower']),
                        'pmr_ci_upper': float(estimates_dict['pmr_ci_upper']),
                        'pmr_ci_cover': bool(estimates_dict['pmr_ci_cover']),
                        'pmr_ci_width': float(estimates_dict['pmr_ci_width']),
                        'por_mse': float((estimates_dict['por'] - true_psi) ** 2),
                        'pipw_mse': float((estimates_dict['pipw'] - true_psi) ** 2),
                        'phe1_mse': float((estimates_dict['phe1'] - true_psi) ** 2),
                        'phe2_mse': float((estimates_dict['phe2'] - true_psi) ** 2),
                        'pmr_mse': float((estimates_dict['pmr'] - true_psi) ** 2),
                        'experiment_index': exp_idx,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    all_results.append(result_record)
                    
                    if (exp_idx + 1) % 10 == 0 or (exp_idx + 1) == config['n_experiments']:
                        pd.DataFrame(all_results).to_csv(csv_filename, index=False)
                        
                except Exception as e:
                    error_record = {
                        'group': group_name, 'sample_size': sample_size, 'seed': seed,
                        'experiment_index': exp_idx, 'error': str(e),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    print(f"\nWarning: Experiment {exp_idx} (seed={seed}) failed: {str(e)}")
                    error_filename = os.path.join(group_dir, f"n{sample_size}_errors.csv")
                    error_df = pd.DataFrame([error_record])
                    if os.path.exists(error_filename):
                        error_df.to_csv(error_filename, mode='a', header=False, index=False)
                    else:
                        error_df.to_csv(error_filename, index=False)
            
            if all_results:
                print(f"\nSample size {sample_size} completed! Number of successful experiments: {len(all_results)}")
                df = pd.DataFrame(all_results)
                if 'pmr_ci_cover' in df.columns:
                    print(f"PMR 95% confidence interval coverage: {df['pmr_ci_cover'].mean() *100:.2f}%")
                    print(f"Mean confidence interval width: {df['pmr_ci_width'].mean():.6f}")

    print("\n" + "="*80)
    print("All experiments completed! Summary report is being generated...")
    create_summary_report_with_ci(results_dir)

def create_summary_report_with_ci(results_dir):
    """Create an experiment summary report with confidence interval statistics"""
    import glob
    
    result_files = glob.glob(os.path.join(results_dir, "**", "*_results.csv"), recursive=True)
    if not result_files:
        Print("Result file not found")
        return
        
    all_dfs = []
    for file in result_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Failed to read file {file}: {e}")
            
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_file = os.path.join(results_dir, "all_experiments_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        
        summary_stats = []
        for (group, sample_size), group_df in combined_df.groupby(['group', 'sample_size']):
            summary = {
                'group': group, 'sample_size': sample_size, 'n_experiments': len(group_df),
                'true_mean': group_df['true_psi'].mean(), 'true_std': group_df['true_psi'].std(),
            }
            if 'pmr_ci_cover' in group_df.columns:
                summary['pmr_ci_coverage'] = group_df['pmr_ci_cover'].mean() * 100
                summary['pmr_ci_width_mean'] = group_df['pmr_ci_width'].mean()
                summary['pmr_se_mean'] = group_df['pmr_se'].mean()
                
            # Loop through the five estimators to extract mse and bias
            for estimator in ['por', 'pipw', 'phe1', 'phe2', 'pmr']:
                mse_col = f'{estimator}_mse'
                if mse_col in group_df.columns:
                    summary[f'{estimator}_mse_mean'] = group_df[mse_col].mean()
                est_col = f'{estimator}_estimate'
                if est_col in group_df.columns:
                    summary[f'{estimator}_bias_mean'] = (group_df[est_col] - group_df['true_psi']).mean()
            summary_stats.append(summary)
            
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(results_dir, "experiment_summary_with_ci.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print("\nSummary statistics (with confidence intervals):")
        print(summary_df.to_string())
        
if __name__ == "__main__":
    main()