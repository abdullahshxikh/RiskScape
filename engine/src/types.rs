use serde::{Deserialize, Serialize};

/// Represents historical returns for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetReturn {
    /// Asset name or identifier
    pub name: String,
    /// Time series of asset returns
    pub returns: Vec<f64>,
}

/// Represents historical returns for a factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorReturn {
    /// Factor name (e.g., "Market", "Value", "Growth")
    pub name: String,
    /// Time series of factor returns
    pub returns: Vec<f64>,
}

/// Results from multi-factor regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Intercept term (alpha) of the regression
    pub alpha: f64,
    /// Beta coefficients for each factor (factor name, beta value)
    pub betas: Vec<(String, f64)>,
    /// R-squared value indicating goodness of fit
    pub r_squared: f64,
    /// Residuals from the regression
    pub residuals: Vec<f64>,
    /// Fitted values from the regression
    pub fitted_values: Vec<f64>,
}

/// Covariance matrix results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovarianceResult {
    /// Covariance matrix as 2D vector
    pub matrix: Vec<Vec<f64>>,
    /// Variable names corresponding to matrix rows/columns
    pub variables: Vec<String>,
}

/// Principal Component Analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCAResult {
    /// Eigenvalues representing variance explained by each component
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (loadings) for each principal component
    pub eigenvectors: Vec<Vec<f64>>,
    /// Explained variance ratio for each component
    pub explained_variance_ratio: Vec<f64>,
    /// Cumulative explained variance ratio
    pub cumulative_variance_ratio: Vec<f64>,
}

/// Stress test results showing shocked portfolio metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Scenario results as HashMap of scenario name to loss distribution
    pub scenario_results: std::collections::HashMap<String, Vec<f64>>,
    /// Loss distribution under stress scenarios
    pub loss_distribution: Vec<f64>,
    /// Value-at-Risk at 95% confidence level
    pub var_95: f64,
    /// Value-at-Risk at 99% confidence level
    pub var_99: f64,
    /// Expected Shortfall (Conditional VaR)
    pub expected_shortfall: f64,
}

/// Risk decomposition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskDecompositionResult {
    /// Total portfolio risk (volatility)
    pub total_risk: f64,
    /// Systematic risk (explained by market factors)
    pub systematic_risk: f64,
    /// Idiosyncratic risk (asset-specific risk)
    pub idiosyncratic_risk: f64,
    /// Factor contributions to risk
    pub factor_contributions: Vec<f64>,
    /// Factor names
    pub factor_names: Vec<String>,
}

/// Input data structure for analysis functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisInput {
    /// Asset return data
    pub assets: Vec<AssetReturn>,
    /// Factor return data (optional, used in multi-factor models)
    pub factors: Option<Vec<FactorReturn>>,
    /// Analysis parameters (confidence levels, scenarios, etc.)
    pub parameters: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Output structure for analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOutput<T> {
    /// Whether the analysis was successful
    pub success: bool,
    /// The analysis result (if successful)
    pub result: Option<T>,
    /// Error message (if analysis failed)
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub execution_time_ms: Option<u64>,
}

/// Portfolio composition and weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Asset weights (should sum to 1.0)
    pub weights: Vec<f64>,
    /// Asset names corresponding to weights
    pub asset_names: Vec<String>,
}

/// Risk metrics for a portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Portfolio volatility (standard deviation)
    pub volatility: f64,
    /// Value at Risk at 95% confidence
    pub var_95: f64,
    /// Value at Risk at 99% confidence
    pub var_99: f64,
    /// Expected Shortfall (Conditional VaR)
    pub expected_shortfall: f64,
    /// Sharpe ratio (if benchmark return provided)
    pub sharpe_ratio: Option<f64>,
    /// Maximum drawdown
    pub max_drawdown: f64,
}

/// Factor model specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorModel {
    /// Model type (e.g., "CAPM", "Fama-French", "Custom")
    pub model_type: String,
    /// Factor names included in the model
    pub factor_names: Vec<String>,
    /// Risk-free rate for Sharpe ratio calculation
    pub risk_free_rate: Option<f64>,
}
