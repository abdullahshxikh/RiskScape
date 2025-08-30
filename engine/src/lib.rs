mod types;
mod model;

use wasm_bindgen::prelude::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use types::*;
use rand::prelude::*;
use rand_pcg::Pcg64;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, riskscape!");
}



#[derive(Serialize, Deserialize)]
pub struct CovarianceMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub variables: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct StressTestResult {
    pub scenario_results: HashMap<String, Vec<f64>>,
    pub loss_distribution: Vec<f64>,
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall: f64,
}

#[derive(Serialize, Deserialize)]
pub struct RiskDecompositionResult {
    pub total_risk: f64,
    pub systematic_risk: f64,
    pub idiosyncratic_risk: f64,
    pub factor_contributions: Vec<f64>,
    pub factor_names: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SyntheticDataConfig {
    pub n_factors: usize,
    pub n_assets: usize,
    pub n_periods: usize,
    pub factor_volatility: Vec<f64>,
    pub asset_volatility: Vec<f64>,
    pub correlations: Vec<Vec<f64>>, // Factor correlations
}

#[derive(Serialize, Deserialize)]
pub struct AutoSimResult {
    pub factors: Vec<FactorReturn>,
    pub assets: Vec<AssetReturn>,
    pub regression_results: Vec<RegressionResult>,
    pub pca_result: PCAResult,
    pub timestamp: f64,
}

#[derive(Serialize, Deserialize)]
pub struct AnalysisInput {
    pub data: Vec<Vec<f64>>,
    pub variable_names: Option<Vec<String>>,
    pub parameters: Option<HashMap<String, f64>>,
}

#[derive(Serialize, Deserialize)]
pub struct AnalysisOutput {
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

#[wasm_bindgen]
pub struct RiskEngine {
    data: Option<DMatrix<f64>>,
}

#[wasm_bindgen]
impl RiskEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RiskEngine {
        RiskEngine { data: None }
    }

    #[wasm_bindgen]
    pub fn load_data(&mut self, data: &[f64], rows: usize, cols: usize) {
        let matrix_data: Vec<f64> = data.to_vec();
        self.data = Some(DMatrix::from_row_slice(rows, cols, &matrix_data));
    }

    #[wasm_bindgen]
    pub fn load_data_from_json(&mut self, json_input: &str) -> Result<String, JsValue> {
        match serde_json::from_str::<AnalysisInput>(json_input) {
            Ok(input) => {
                if input.data.is_empty() || input.data[0].is_empty() {
                    return Ok(serde_json::to_string(&AnalysisOutput {
                        success: false,
                        result: None,
                        error: Some("Empty data provided".to_string()),
                    }).unwrap());
                }

                let rows = input.data.len();
                let cols = input.data[0].len();

                // Flatten the 2D vector into 1D
                let mut flat_data = Vec::with_capacity(rows * cols);
                for row in &input.data {
                    if row.len() != cols {
                        return Ok(serde_json::to_string(&AnalysisOutput {
                            success: false,
                            result: None,
                            error: Some("Inconsistent row lengths".to_string()),
                        }).unwrap());
                    }
                    flat_data.extend_from_slice(row);
                }

                self.data = Some(DMatrix::from_row_slice(rows, cols, &flat_data));

                Ok(serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::json!({
                        "rows": rows,
                        "cols": cols,
                        "loaded": true
                    })),
                    error: None,
                }).unwrap())
            }
            Err(e) => Ok(serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("JSON parsing error: {}", e)),
            }).unwrap())
        }
    }

    #[wasm_bindgen]
    pub fn linear_regression(&self, y_col: usize, x_cols: &[usize]) -> Result<JsValue, JsValue> {
        if let Some(data) = &self.data {
            if y_col >= data.ncols() || x_cols.iter().any(|&col| col >= data.ncols()) {
                return Err(JsValue::from_str("Column index out of bounds"));
            }

            let y = data.column(y_col);
            let mut _x_data: Vec<f64> = Vec::new();
            for &col in x_cols {
                _x_data.extend(data.column(col).iter().cloned());
            }

            let n = data.nrows();
            let p = x_cols.len();

            // Add intercept column (ones)
            let mut x_matrix = DMatrix::zeros(n, p + 1);
            x_matrix.column_mut(0).fill(1.0);
            for (i, &col) in x_cols.iter().enumerate() {
                x_matrix.column_mut(i + 1).copy_from(&data.column(col));
            }

            // Calculate coefficients using normal equation: (X^T X)^-1 X^T y
            let xt = x_matrix.transpose();
            let xtx = &xt * &x_matrix;
            let xty = &xt * y;

            let coefficients_vec = match xtx.try_inverse() {
                Some(xtx_inv) => (xtx_inv * xty).data.as_vec().clone(),
                None => return Err(JsValue::from_str("Matrix is singular, cannot invert")),
            };

            // Calculate fitted values and residuals
            let fitted = &x_matrix * DVector::from_vec(coefficients_vec.clone());
            let fitted_values = fitted.data.as_vec().clone();
            let residuals: Vec<f64> = (y - fitted).iter().cloned().collect();

            // Calculate R-squared
            let y_mean = y.mean();
            let ss_tot: f64 = y.iter().map(|&val| (val - y_mean).powi(2)).sum();
            let ss_res: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
            let r_squared = 1.0 - (ss_res / ss_tot);

            // Extract alpha (intercept) and betas
            let alpha = coefficients_vec[0];
            let betas: Vec<(String, f64)> = x_cols.iter()
                .enumerate()
                .map(|(i, &col)| {
                    let col_name = if col < data.ncols() {
                        format!("Column_{}", col)
                    } else {
                        format!("Var_{}", col)
                    };
                    (col_name, coefficients_vec[i + 1])
                })
                .collect();

            let result = types::RegressionResult {
                alpha,
                betas,
                r_squared,
                residuals,
                fitted_values,
            };

            Ok(serde_wasm_bindgen::to_value(&result)?)
        } else {
            Err(JsValue::from_str("No data loaded"))
        }
    }

    #[wasm_bindgen]
    pub fn principal_component_analysis(&self) -> Result<JsValue, JsValue> {
        if let Some(data) = &self.data {
            // Center the data
            let means: Vec<f64> = (0..data.ncols()).map(|col| data.column(col).mean()).collect();
            let mut centered_data = data.clone();

            for col in 0..data.ncols() {
                let mean = means[col];
                centered_data.column_mut(col).iter_mut().for_each(|x| *x -= mean);
            }

            // Calculate covariance matrix
            let cov = (&centered_data.transpose() * &centered_data) / (centered_data.nrows() as f64 - 1.0);

            // Eigen decomposition
            let eigen = SymmetricEigen::new(cov);

            let eigenvalues: Vec<f64> = eigen.eigenvalues.data.as_vec().clone();
            let eigenvectors: Vec<Vec<f64>> = eigen.eigenvectors
                .column_iter()
                .map(|col| col.iter().cloned().collect())
                .collect();

            // Calculate explained variance ratios
            let total_variance: f64 = eigenvalues.iter().sum();
            let explained_variance_ratio: Vec<f64> = eigenvalues.iter()
                .map(|&ev| ev / total_variance)
                .collect();

            let mut cumulative = 0.0;
            let cumulative_variance_ratio: Vec<f64> = explained_variance_ratio.iter()
                .map(|&ratio| {
                    cumulative += ratio;
                    cumulative
                })
                .collect();

            let result = types::PCAResult {
                eigenvalues,
                eigenvectors,
                explained_variance_ratio,
                cumulative_variance_ratio,
            };

            Ok(serde_wasm_bindgen::to_value(&result)?)
        } else {
            Err(JsValue::from_str("No data loaded"))
        }
    }

    #[wasm_bindgen]
    pub fn calculate_covariance_matrix(&self) -> Result<JsValue, JsValue> {
        if let Some(data) = &self.data {
            let n = data.nrows() as f64;
            let means: Vec<f64> = (0..data.ncols()).map(|col| data.column(col).mean()).collect();

            let mut cov_matrix = DMatrix::zeros(data.ncols(), data.ncols());

            for i in 0..data.ncols() {
                for j in 0..data.ncols() {
                    let cov: f64 = (0..data.nrows())
                        .map(|row| (data[(row, i)] - means[i]) * (data[(row, j)] - means[j]))
                        .sum::<f64>() / (n - 1.0);
                    cov_matrix[(i, j)] = cov;
                }
            }

            let matrix: Vec<Vec<f64>> = cov_matrix.row_iter()
                .map(|row| row.iter().cloned().collect())
                .collect();

            let variables: Vec<String> = (0..data.ncols())
                .map(|i| format!("Variable_{}", i))
                .collect();

            let result = CovarianceMatrix { matrix, variables };

            Ok(serde_wasm_bindgen::to_value(&result)?)
        } else {
            Err(JsValue::from_str("No data loaded"))
        }
    }

    #[wasm_bindgen]
    pub fn run_stress_test(&self, scenarios: &[f64], confidence_levels: &[f64]) -> Result<JsValue, JsValue> {
        if let Some(_data) = &self.data {
            use rand::prelude::*;
            use rand_pcg::Pcg64;

            let mut rng = Pcg64::from_entropy();

            // Generate scenario results (simplified Monte Carlo)
            let n_scenarios = 1000;
            let mut scenario_results: HashMap<String, Vec<f64>> = HashMap::new();
            let mut loss_distribution = Vec::with_capacity(n_scenarios);

            for &scenario in scenarios {
                let mut results = Vec::with_capacity(n_scenarios);
                for _ in 0..n_scenarios {
                    // Simplified stress test simulation
                    let shock = scenario + rng.gen::<f64>() * 0.1;
                    let loss = shock * rng.gen::<f64>();
                    results.push(loss);
                    if scenario_results.len() == 0 {
                        loss_distribution.push(loss);
                    }
                }
                scenario_results.insert(format!("Scenario_{:.2}", scenario), results);
            }

            // Calculate VaR and Expected Shortfall
            loss_distribution.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let var_95_idx = ((1.0 - confidence_levels[0]) * n_scenarios as f64) as usize;
            let var_99_idx = ((1.0 - confidence_levels[1]) * n_scenarios as f64) as usize;

            let var_95 = loss_distribution[var_95_idx];
            let var_99 = loss_distribution[var_99_idx];

            let tail_losses: Vec<f64> = loss_distribution.iter()
                .skip(var_95_idx)
                .cloned()
                .collect();
            let expected_shortfall = tail_losses.iter().sum::<f64>() / tail_losses.len() as f64;

            let result = StressTestResult {
                scenario_results,
                loss_distribution,
                var_95,
                var_99,
                expected_shortfall,
            };

            Ok(serde_wasm_bindgen::to_value(&result)?)
        } else {
            Err(JsValue::from_str("No data loaded"))
        }
    }

    #[wasm_bindgen]
    pub fn risk_decomposition(&self) -> Result<JsValue, JsValue> {
        if let Some(data) = &self.data {
            if data.ncols() < 2 {
                return Err(JsValue::from_str("Need at least 2 variables for risk decomposition"));
            }

            // Perform PCA to identify principal components
            let pca_result = self.principal_component_analysis()?;
            let pca: PCAResult = serde_wasm_bindgen::from_value(pca_result)?;

            // Calculate total portfolio risk (simplified as sum of variances)
            let total_risk: f64 = data.column_iter()
                .map(|col| {
                    let mean = col.mean();
                    col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (col.len() as f64 - 1.0)
                })
                .sum::<f64>()
                .sqrt();

            // Systematic risk (explained by first few principal components)
            let n_systematic = (pca.eigenvalues.len() as f64 * 0.8).ceil() as usize; // 80% variance
            let systematic_risk = pca.eigenvalues.iter()
                .take(n_systematic)
                .sum::<f64>()
                .sqrt();

            // Idiosyncratic risk (remaining unexplained variance)
            let idiosyncratic_risk = (total_risk.powi(2) - systematic_risk.powi(2)).sqrt();

            // Factor contributions (eigenvalues as percentages)
            let total_eigen = pca.eigenvalues.iter().sum::<f64>();
            let factor_contributions: Vec<f64> = pca.eigenvalues.iter()
                .map(|&ev| ev / total_eigen)
                .collect();

            let factor_names: Vec<String> = (0..pca.eigenvalues.len())
                .map(|i| format!("Factor {}", i + 1))
                .collect();

            let result = RiskDecompositionResult {
                total_risk,
                systematic_risk,
                idiosyncratic_risk,
                factor_contributions,
                factor_names,
            };

            Ok(serde_wasm_bindgen::to_value(&result)?)
        } else {
            Err(JsValue::from_str("No data loaded"))
        }
    }

    // JSON-based versions of analysis functions
    #[wasm_bindgen]
    pub fn linear_regression_json(&self, params_json: &str) -> String {
        match serde_json::from_str::<HashMap<String, serde_json::Value>>(params_json) {
            Ok(params) => {
                let y_col = params.get("y_col")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let x_cols: Vec<usize> = params.get("x_cols")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect()
                    )
                    .unwrap_or_default();

                match self.linear_regression(y_col, &x_cols) {
                    Ok(result) => {
                        let regression: RegressionResult = serde_wasm_bindgen::from_value(result).unwrap();
                        serde_json::to_string(&AnalysisOutput {
                            success: true,
                            result: Some(serde_json::to_value(regression).unwrap()),
                            error: None,
                        }).unwrap()
                    }
                    Err(e) => serde_json::to_string(&AnalysisOutput {
                        success: false,
                        result: None,
                        error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
                    }).unwrap()
                }
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Parameter parsing error: {}", e)),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn pca_json(&self) -> String {
        match self.principal_component_analysis() {
            Ok(result) => {
                let pca: PCAResult = serde_wasm_bindgen::from_value(result).unwrap();
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(pca).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn covariance_json(&self) -> String {
        match self.calculate_covariance_matrix() {
            Ok(result) => {
                let cov: CovarianceMatrix = serde_wasm_bindgen::from_value(result).unwrap();
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(cov).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn stress_test_json(&self, params_json: &str) -> String {
        match serde_json::from_str::<HashMap<String, serde_json::Value>>(params_json) {
            Ok(params) => {
                let scenarios: Vec<f64> = params.get("scenarios")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect()
                    )
                    .unwrap_or(vec![1.0, 2.0, 3.0]);

                let confidence_levels: Vec<f64> = params.get("confidence_levels")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect()
                    )
                    .unwrap_or(vec![0.95, 0.99]);

                match self.run_stress_test(&scenarios, &confidence_levels) {
                    Ok(result) => {
                        let stress: StressTestResult = serde_wasm_bindgen::from_value(result).unwrap();
                        serde_json::to_string(&AnalysisOutput {
                            success: true,
                            result: Some(serde_json::to_value(stress).unwrap()),
                            error: None,
                        }).unwrap()
                    }
                    Err(e) => serde_json::to_string(&AnalysisOutput {
                        success: false,
                        result: None,
                        error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
                    }).unwrap()
                }
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Parameter parsing error: {}", e)),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn risk_decomposition_json(&self) -> String {
        match self.risk_decomposition() {
            Ok(result) => {
                let decomp: RiskDecompositionResult = serde_wasm_bindgen::from_value(result).unwrap();
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(decomp).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
            }).unwrap()
        }
    }

    // Model functions exposed to WASM

    #[wasm_bindgen]
    pub fn model_run_regression(&self, asset_json: &str, factors_json: &str) -> String {
        match (serde_json::from_str::<AssetReturn>(asset_json),
               serde_json::from_str::<Vec<FactorReturn>>(factors_json)) {
            (Ok(asset), Ok(factors)) => {
                let result = model::run_regression(asset, factors);
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(result).unwrap()),
                    error: None,
                }).unwrap()
            }
            (Err(e), _) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Asset parsing error: {}", e)),
            }).unwrap(),
            (_, Err(e)) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Factors parsing error: {}", e)),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn model_compute_covariance(&self, assets_json: &str) -> String {
        match serde_json::from_str::<Vec<AssetReturn>>(assets_json) {
            Ok(assets) => {
                let result = model::compute_covariance(assets);
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(result).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Assets parsing error: {}", e)),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn model_run_pca(&self, cov_json: &str) -> String {
        match serde_json::from_str::<CovarianceResult>(cov_json) {
            Ok(cov) => {
                let result = model::run_pca(cov);
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(result).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Covariance parsing error: {}", e)),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn model_stress_test(&self, cov_json: &str, shocks_json: &str) -> String {
        match (serde_json::from_str::<CovarianceResult>(cov_json),
               serde_json::from_str::<Vec<(String, f64)>>(shocks_json)) {
            (Ok(cov), Ok(shocks)) => {
                let result = model::stress_test(cov, shocks);
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(result).unwrap()),
                    error: None,
                }).unwrap()
            }
            (Err(e), _) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Covariance parsing error: {}", e)),
            }).unwrap(),
            (_, Err(e)) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(format!("Shocks parsing error: {}", e)),
            }).unwrap()
        }
    }

    // Direct JSON-to-JsValue functions for easier JavaScript integration

    #[wasm_bindgen]
    pub fn regression(&self, asset_json: &str, factors_json: &str) -> Result<JsValue, JsValue> {
        // Parse JSON inputs into structs
        let asset: AssetReturn = serde_json::from_str(asset_json)
            .map_err(|e| JsValue::from_str(&format!("Asset JSON parsing error: {}", e)))?;

        let factors: Vec<FactorReturn> = serde_json::from_str(factors_json)
            .map_err(|e| JsValue::from_str(&format!("Factors JSON parsing error: {}", e)))?;

        // Run computation
        let result = model::run_regression(asset, factors);

        // Serialize result to JsValue
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Result serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn covariance(&self, assets_json: &str) -> Result<JsValue, JsValue> {
        // Parse JSON input into struct
        let assets: Vec<AssetReturn> = serde_json::from_str(assets_json)
            .map_err(|e| JsValue::from_str(&format!("Assets JSON parsing error: {}", e)))?;

        // Run computation
        let result = model::compute_covariance(assets);

        // Serialize result to JsValue
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Result serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn pca(&self, cov_json: &str) -> Result<JsValue, JsValue> {
        // Parse JSON input into struct
        let cov: CovarianceResult = serde_json::from_str(cov_json)
            .map_err(|e| JsValue::from_str(&format!("Covariance JSON parsing error: {}", e)))?;

        // Run computation
        let result = model::run_pca(cov);

        // Serialize result to JsValue
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Result serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn stress(&self, cov_json: &str, shocks_json: &str) -> Result<JsValue, JsValue> {
        // Parse JSON inputs into structs
        let cov: CovarianceResult = serde_json::from_str(cov_json)
            .map_err(|e| JsValue::from_str(&format!("Covariance JSON parsing error: {}", e)))?;

        let shocks: Vec<(String, f64)> = serde_json::from_str(shocks_json)
            .map_err(|e| JsValue::from_str(&format!("Shocks JSON parsing error: {}", e)))?;

        // Run computation
        let result = model::stress_test(cov, shocks);

        // Serialize result to JsValue
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Result serialization error: {}", e)))
    }

    // Auto-sim mode functions
    #[wasm_bindgen]
    pub fn generate_synthetic_factors(&self, config_json: &str) -> Result<JsValue, JsValue> {
        let config: SyntheticDataConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {}", e)))?;

        let mut rng = Pcg64::from_entropy();
        let mut factors = Vec::with_capacity(config.n_factors);

        // Generate correlated factor returns
        for i in 0..config.n_factors {
            let mut returns = Vec::with_capacity(config.n_periods);

            // Start with a base return
            let mut current_return = 0.0;

            for _ in 0..config.n_periods {
                // Add correlation with other factors
                let mut correlated_noise = 0.0;
                for j in 0..config.n_factors {
                    if i != j && j < config.correlations.len() && i < config.correlations[j].len() {
                        correlated_noise += rng.sample::<f64, _>(rand_distr::StandardNormal) * config.correlations[j][i].sqrt();
                    }
                }

                // Generate return with volatility and correlation
                let volatility = if i < config.factor_volatility.len() { config.factor_volatility[i] } else { 0.02 };
                let noise = rng.sample::<f64, _>(rand_distr::StandardNormal);
                let return_change = volatility * (0.7 * noise + 0.3 * correlated_noise / config.n_factors as f64);

                current_return += return_change;
                returns.push(current_return);
            }

            factors.push(FactorReturn {
                name: format!("Factor_{}", i + 1),
                returns,
            });
        }

        serde_wasm_bindgen::to_value(&factors)
            .map_err(|e| JsValue::from_str(&format!("Factors serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn generate_synthetic_assets(&self, factors_json: &str, config_json: &str) -> Result<JsValue, JsValue> {
        let factors: Vec<FactorReturn> = serde_json::from_str(factors_json)
            .map_err(|e| JsValue::from_str(&format!("Factors parsing error: {}", e)))?;

        let config: SyntheticDataConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parsing error: {}", e)))?;

        let mut rng = Pcg64::from_entropy();
        let mut assets = Vec::with_capacity(config.n_assets);
        let n_periods = factors[0].returns.len();

        for i in 0..config.n_assets {
            let mut returns = Vec::with_capacity(n_periods);

            // Generate random factor exposures (betas)
            let mut betas = Vec::with_capacity(factors.len());
            for _ in 0..factors.len() {
                betas.push(rng.gen::<f64>() * 2.0 - 1.0); // Random beta between -1 and 1
            }

            // Generate asset-specific alpha
            let alpha = (rng.gen::<f64>() - 0.5) * 0.02; // Small random alpha

            for t in 0..n_periods {
                let mut expected_return = alpha;

                // Add factor contributions
                for (j, factor) in factors.iter().enumerate() {
                    expected_return += betas[j] * factor.returns[t];
                }

                // Add asset-specific volatility and random noise
                let volatility = if i < config.asset_volatility.len() { config.asset_volatility[i] } else { 0.03 };
                let noise = rng.sample::<f64, _>(rand_distr::StandardNormal);
                let idiosyncratic_return = volatility * noise;

                returns.push(expected_return + idiosyncratic_return);
            }

            assets.push(AssetReturn {
                name: format!("Asset_{}", i + 1),
                returns,
            });
        }

        serde_wasm_bindgen::to_value(&assets)
            .map_err(|e| JsValue::from_str(&format!("Assets serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn run_auto_sim_analysis(&self, factors_json: &str, assets_json: &str) -> Result<JsValue, JsValue> {
        let factors: Vec<FactorReturn> = serde_json::from_str(factors_json)
            .map_err(|e| JsValue::from_str(&format!("Factors parsing error: {}", e)))?;

        let assets: Vec<AssetReturn> = serde_json::from_str(assets_json)
            .map_err(|e| JsValue::from_str(&format!("Assets parsing error: {}", e)))?;

        // Run regression for each asset
        let mut regression_results = Vec::with_capacity(assets.len());
        for asset in &assets {
            let result = model::run_regression(asset.clone(), factors.clone());
            regression_results.push(result);
        }

        // Compute covariance matrix
        let cov_result = model::compute_covariance(assets.clone());

        // Run PCA
        let pca_result = model::run_pca(cov_result);

        let result = AutoSimResult {
            factors,
            assets,
            regression_results,
            pca_result,
            timestamp: js_sys::Date::now(),
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Auto-sim result serialization error: {}", e)))
    }

    // JSON-based auto-sim functions
    #[wasm_bindgen]
    pub fn generate_synthetic_factors_json(&self, config_json: &str) -> String {
        match self.generate_synthetic_factors(config_json) {
            Ok(result) => {
                let factors: Vec<FactorReturn> = serde_wasm_bindgen::from_value(result).unwrap();
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(factors).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn generate_synthetic_assets_json(&self, factors_json: &str, config_json: &str) -> String {
        match self.generate_synthetic_assets(factors_json, config_json) {
            Ok(result) => {
                let assets: Vec<AssetReturn> = serde_wasm_bindgen::from_value(result).unwrap();
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(assets).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
            }).unwrap()
        }
    }

    #[wasm_bindgen]
    pub fn run_auto_sim_analysis_json(&self, factors_json: &str, assets_json: &str) -> String {
        match self.run_auto_sim_analysis(factors_json, assets_json) {
            Ok(result) => {
                let auto_sim: AutoSimResult = serde_wasm_bindgen::from_value(result).unwrap();
                serde_json::to_string(&AnalysisOutput {
                    success: true,
                    result: Some(serde_json::to_value(auto_sim).unwrap()),
                    error: None,
                }).unwrap()
            }
            Err(e) => serde_json::to_string(&AnalysisOutput {
                success: false,
                result: None,
                error: Some(e.as_string().unwrap_or("Unknown error".to_string())),
            }).unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[test]
    fn test_risk_engine_creation() {
        let engine = RiskEngine::new();
        assert!(engine.data.is_none());
    }

    #[test]
    fn test_data_loading() {
        let mut engine = RiskEngine::new();
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        engine.load_data(&test_data, 2, 3);
        assert!(engine.data.is_some());
        if let Some(data) = &engine.data {
            assert_eq!(data.nrows(), 2);
            assert_eq!(data.ncols(), 3);
        }
    }

    #[wasm_bindgen_test]
    fn test_wasm_regression_binding() {
        let engine = RiskEngine::new();

        let asset_json = r#"{
            "name": "TestAsset",
            "returns": [0.01, 0.02, -0.01, 0.03, -0.02]
        }"#;

        let factors_json = r#"[
            {
                "name": "Market",
                "returns": [0.015, 0.025, -0.005, 0.035, -0.015]
            },
            {
                "name": "Value",
                "returns": [0.005, -0.01, 0.02, 0.01, -0.005]
            }
        ]"#;

        let result = engine.regression(asset_json, factors_json);

        match result {
            Ok(js_value) => {
                // The result should be a JsValue containing the RegressionResult
                // We can't easily test the exact content in this context,
                // but we can verify the function doesn't panic and returns a value
                assert!(js_value.is_object());
            }
            Err(e) => panic!("Regression binding failed: {:?}", e),
        }
    }

    #[wasm_bindgen_test]
    fn test_wasm_covariance_binding() {
        let engine = RiskEngine::new();

        let assets_json = r#"[
            {
                "name": "Asset1",
                "returns": [0.01, 0.02, -0.01, 0.03]
            },
            {
                "name": "Asset2",
                "returns": [0.015, 0.025, -0.005, 0.035]
            }
        ]"#;

        let result = engine.covariance(assets_json);

        match result {
            Ok(js_value) => {
                assert!(js_value.is_object());
            }
            Err(e) => panic!("Covariance binding failed: {:?}", e),
        }
    }
}
