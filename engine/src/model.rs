use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::collections::HashMap;

use crate::types::*;

/// Run Ordinary Least Squares regression of asset returns on factor returns
///
/// # Arguments
/// * `asset` - Asset return data with historical returns
/// * `factors` - Vector of factor return data
///
/// # Returns
/// RegressionResult containing alpha (intercept), betas (coefficients), R², residuals, and fitted values
///
/// # Panics
/// Panics if asset and factors have different lengths or if matrix inversion fails
pub fn run_regression(asset: AssetReturn, factors: Vec<FactorReturn>) -> RegressionResult {
    // Validate input data
    let n = asset.returns.len();
    if factors.is_empty() {
        panic!("At least one factor must be provided");
    }

    for factor in &factors {
        if factor.returns.len() != n {
            panic!("All factor return series must have the same length as asset returns");
        }
    }

    // Prepare the design matrix X (factors) and response vector y (asset returns)
    let p = factors.len(); // number of factors
    let mut x_data = Vec::with_capacity(n * (p + 1)); // +1 for intercept

    // Add intercept column (ones)
    for _ in 0..n {
        x_data.push(1.0);
    }

    // Add factor columns
    for i in 0..n {
        for factor in &factors {
            x_data.push(factor.returns[i]);
        }
    }

    // Create matrices
    let x = DMatrix::from_row_slice(n, p + 1, &x_data);
    let y = DVector::from_vec(asset.returns.clone());

    // Compute coefficients using normal equation: β = (X^T X)^(-1) X^T y
    let xt = x.transpose();
    let xtx = &xt * &x;
    let xty = &xt * &y;

    let xtx_inv = match xtx.try_inverse() {
        Some(inv) => inv,
        None => panic!("Matrix X^T X is singular and cannot be inverted"),
    };

    let beta_vec = xtx_inv * xty;

    // Extract alpha (intercept) and betas
    let alpha = beta_vec[0];
    let betas: Vec<(String, f64)> = factors.iter()
        .enumerate()
        .map(|(i, factor)| (factor.name.clone(), beta_vec[i + 1]))
        .collect();

    // Compute fitted values and residuals
    let fitted_values_vec = &x * &beta_vec;
    let residuals: Vec<f64> = (y.clone() - fitted_values_vec.clone()).iter().cloned().collect();

    // Compute R-squared
    let y_mean = y.mean();
    let ss_tot: f64 = y.iter().map(|&val| (val - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
    let r_squared = 1.0 - (ss_res / ss_tot);

    RegressionResult {
        alpha,
        betas,
        r_squared,
        residuals,
        fitted_values: fitted_values_vec.data.as_vec().clone(),
    }
}

/// Compute sample covariance matrix from asset returns
///
/// # Arguments
/// * `assets` - Vector of asset return data
///
/// # Returns
/// CovarianceResult containing the covariance matrix and variable names
///
/// # Panics
/// Panics if assets vector is empty or assets have different lengths
pub fn compute_covariance(assets: Vec<AssetReturn>) -> CovarianceResult {
    if assets.is_empty() {
        panic!("At least one asset must be provided");
    }

    let n_assets = assets.len();
    let n_periods = assets[0].returns.len();

    // Validate that all assets have the same number of periods
    for asset in &assets {
        if asset.returns.len() != n_periods {
            panic!("All assets must have the same number of return periods");
        }
    }

    let mut cov_matrix = vec![vec![0.0; n_assets]; n_assets];
    let mut means = vec![0.0; n_assets];

    // Compute means
    for (i, asset) in assets.iter().enumerate() {
        means[i] = asset.returns.iter().sum::<f64>() / n_periods as f64;
    }

    // Compute covariance matrix
    for i in 0..n_assets {
        for j in 0..n_assets {
            let mut cov = 0.0;
            for t in 0..n_periods {
                cov += (assets[i].returns[t] - means[i]) * (assets[j].returns[t] - means[j]);
            }
            cov_matrix[i][j] = cov / (n_periods - 1) as f64;
        }
    }

    let variables: Vec<String> = assets.iter().map(|a| a.name.clone()).collect();

    CovarianceResult {
        matrix: cov_matrix,
        variables,
    }
}

/// Run Principal Component Analysis on covariance matrix
///
/// # Arguments
/// * `cov` - CovarianceResult containing the covariance matrix
///
/// # Returns
/// PCAResult containing eigenvalues, eigenvectors, and explained variance ratios
///
/// # Panics
/// Panics if covariance matrix is not square or has invalid dimensions
pub fn run_pca(cov: CovarianceResult) -> PCAResult {
    let n = cov.matrix.len();
    if n == 0 {
        panic!("Covariance matrix cannot be empty");
    }

    // Validate that matrix is square
    for row in &cov.matrix {
        if row.len() != n {
            panic!("Covariance matrix must be square");
        }
    }

    // Convert to nalgebra matrix
    let mut matrix_data = Vec::with_capacity(n * n);
    for row in &cov.matrix {
        matrix_data.extend_from_slice(row);
    }

    let cov_matrix = DMatrix::from_row_slice(n, n, &matrix_data);

    // Perform eigen decomposition
    let eigen = SymmetricEigen::new(cov_matrix);

    let eigenvalues: Vec<f64> = eigen.eigenvalues.data.as_vec().clone();
    let eigenvectors: Vec<Vec<f64>> = eigen.eigenvectors
        .column_iter()
        .map(|col| col.iter().cloned().collect())
        .collect();

    // Compute explained variance ratios
    let total_variance: f64 = eigenvalues.iter().sum();
    let explained_variance_ratio: Vec<f64> = eigenvalues.iter()
        .map(|&ev| ev / total_variance)
        .collect();

    // Compute cumulative explained variance ratios
    let mut cumulative = 0.0;
    let cumulative_variance_ratio: Vec<f64> = explained_variance_ratio.iter()
        .map(|&ratio| {
            cumulative += ratio;
            cumulative
        })
        .collect();

    PCAResult {
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
        cumulative_variance_ratio,
    }
}

/// Run stress test by applying shocks to covariance matrix
///
/// # Arguments
/// * `cov` - Original covariance matrix
/// * `shocks` - Vector of (variable_name, shock_factor) pairs
///              shock_factor > 1 increases variance/covariance
///              shock_factor < 1 decreases variance/covariance
///
/// # Returns
/// StressTestResult with shocked covariance matrix and risk metrics
///
/// # Panics
/// Panics if shock factors are invalid or variable names not found
pub fn stress_test(cov: CovarianceResult, shocks: Vec<(String, f64)>) -> StressTestResult {
    // Validate shock factors
    for (_, shock) in &shocks {
        if *shock <= 0.0 {
            panic!("Shock factors must be positive");
        }
    }

    // Create shocked covariance matrix
    let n = cov.matrix.len();
    let mut shocked_matrix = cov.matrix.clone();

    // Apply shocks to variances and covariances
    for (var_name, shock) in &shocks {
        // Find the index of the variable
        let var_idx = match cov.variables.iter().position(|name| name == var_name) {
            Some(idx) => idx,
            None => panic!("Variable '{}' not found in covariance matrix", var_name),
        };

        // Apply shock to variance
        shocked_matrix[var_idx][var_idx] *= shock;

        // Apply shock to covariances (simplified approach)
        for j in 0..n {
            if j != var_idx {
                // Apply shock to off-diagonal elements
                shocked_matrix[var_idx][j] *= shock.sqrt();
                shocked_matrix[j][var_idx] *= shock.sqrt();
            }
        }
    }

    // Simple Monte Carlo simulation for loss distribution
    use rand::prelude::*;
    use rand_pcg::Pcg64;

    let mut rng = Pcg64::from_entropy();
    let n_scenarios = 10000;

    // Convert shocked matrix to nalgebra for Cholesky decomposition
    let mut matrix_data = Vec::with_capacity(n * n);
    for row in &shocked_matrix {
        matrix_data.extend_from_slice(row);
    }
    let cov_matrix = DMatrix::from_row_slice(n, n, &matrix_data);

    // Perform Cholesky decomposition for correlated random variables
    let chol = match cov_matrix.cholesky() {
        Some(c) => c,
        None => panic!("Covariance matrix is not positive definite after applying shocks"),
    };

    let mut loss_distribution = Vec::with_capacity(n_scenarios);
    let mut scenario_results: HashMap<String, Vec<f64>> = HashMap::new();

    // Generate scenarios
    for _ in 0..n_scenarios {
        // Generate uncorrelated normal random variables
        let uncorrelated: Vec<f64> = (0..n).map(|_| rng.sample::<f64, _>(rand_distr::StandardNormal)).collect();

        // Apply Cholesky transformation for correlation
        let correlated = chol.l() * DVector::from_vec(uncorrelated);

        // Simple loss calculation (sum of returns - can be made more sophisticated)
        let loss = correlated.iter().sum::<f64>();
        loss_distribution.push(loss);
    }

    // Sort for VaR calculation
    loss_distribution.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate VaR at different confidence levels
    let var_95_idx = ((1.0 - 0.95) * n_scenarios as f64) as usize;
    let var_99_idx = ((1.0 - 0.99) * n_scenarios as f64) as usize;

    let var_95 = loss_distribution[var_95_idx];
    let var_99 = loss_distribution[var_99_idx];

    // Calculate Expected Shortfall (Conditional VaR)
    let tail_losses: Vec<f64> = loss_distribution.iter()
        .skip(var_95_idx)
        .cloned()
        .collect();
    let expected_shortfall = tail_losses.iter().sum::<f64>() / tail_losses.len() as f64;

    // Create scenario results (simplified - could be more detailed)
    scenario_results.insert("Base Scenario".to_string(), loss_distribution.clone());

    StressTestResult {
        scenario_results,
        loss_distribution,
        var_95,
        var_99,
        expected_shortfall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_regression() {
        // Create test data
        let asset = AssetReturn {
            name: "Test Asset".to_string(),
            returns: vec![0.01, 0.02, -0.01, 0.03, -0.02],
        };

        let factor1 = FactorReturn {
            name: "Market".to_string(),
            returns: vec![0.015, 0.025, -0.005, 0.035, -0.015],
        };

        let factor2 = FactorReturn {
            name: "Value".to_string(),
            returns: vec![0.005, -0.01, 0.02, 0.01, -0.005],
        };

        let result = run_regression(asset, vec![factor1, factor2]);

        assert!(result.alpha.is_finite());
        assert_eq!(result.betas.len(), 2);
        assert!(result.r_squared >= 0.0 && result.r_squared <= 1.0);
        assert_eq!(result.residuals.len(), 5);
        assert_eq!(result.fitted_values.len(), 5);
    }

    #[test]
    fn test_compute_covariance() {
        let asset1 = AssetReturn {
            name: "Asset1".to_string(),
            returns: vec![0.01, 0.02, -0.01, 0.03],
        };

        let asset2 = AssetReturn {
            name: "Asset2".to_string(),
            returns: vec![0.015, 0.025, -0.005, 0.035],
        };

        let result = compute_covariance(vec![asset1, asset2]);

        assert_eq!(result.matrix.len(), 2);
        assert_eq!(result.matrix[0].len(), 2);
        assert_eq!(result.variables.len(), 2);
        assert!(result.matrix[0][0].is_finite()); // variance should be positive
        assert!(result.matrix[1][1].is_finite());
    }

    #[test]
    fn test_run_pca() {
        // Create a simple 2x2 covariance matrix
        let cov = CovarianceResult {
            matrix: vec![
                vec![0.04, 0.02],
                vec![0.02, 0.09],
            ],
            variables: vec!["Var1".to_string(), "Var2".to_string()],
        };

        let result = run_pca(cov);

        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.len(), 2);
        assert_eq!(result.explained_variance_ratio.len(), 2);
        assert_eq!(result.cumulative_variance_ratio.len(), 2);

        // Check that eigenvalues are positive
        for &ev in &result.eigenvalues {
            assert!(ev >= 0.0);
        }

        // Check that explained variance ratios sum to approximately 1
        let sum_ratio: f64 = result.explained_variance_ratio.iter().sum();
        assert!((sum_ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stress_test() {
        let cov = CovarianceResult {
            matrix: vec![
                vec![0.04, 0.02],
                vec![0.02, 0.09],
            ],
            variables: vec!["Asset1".to_string(), "Asset2".to_string()],
        };

        let shocks = vec![
            ("Asset1".to_string(), 1.5), // 50% increase in variance
        ];

        let result = stress_test(cov, shocks);

        assert!(result.loss_distribution.len() > 0);
        assert!(result.var_95.is_finite());
        assert!(result.var_99.is_finite());
        assert!(result.expected_shortfall.is_finite());
        assert!(result.scenario_results.contains_key("Base Scenario"));
    }

    #[test]
    #[should_panic(expected = "At least one factor must be provided")]
    fn test_regression_no_factors() {
        let asset = AssetReturn {
            name: "Test".to_string(),
            returns: vec![0.01, 0.02],
        };

        run_regression(asset, vec![]);
    }

    #[test]
    #[should_panic(expected = "All factor return series must have the same length")]
    fn test_regression_mismatched_lengths() {
        let asset = AssetReturn {
            name: "Test".to_string(),
            returns: vec![0.01, 0.02],
        };

        let factor = FactorReturn {
            name: "Factor".to_string(),
            returns: vec![0.01], // Different length
        };

        run_regression(asset, vec![factor]);
    }
}
