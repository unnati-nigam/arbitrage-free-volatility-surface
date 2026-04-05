# Arbitrage-Free Implied Volatility Surface Construction via Gaussian Processes

**Domain:** Quantitative Finance, Non-Parametric Machine Learning, Derivatives Pricing  
**Tech Stack:** Python, SciPy, Scikit-Learn, Pandas, Matplotlib  

## Executive Summary
This project implements an end-to-end quantitative pipeline to construct implied volatility (IV) surfaces from real-world market options data. It bridges the gap between raw machine learning and theoretical finance by demonstrating how standard ML models (Gaussian Processes) often violate the fundamental laws of no-arbitrage, and implements a mathematical projection engine to fix these violations.

The pipeline is wrapped in a dynamic, Object-Oriented architecture, proving cross-asset robustness across Large-Cap (SPY), Tech/Growth (QQQ), and Small-Cap (IWM) market microstructures.

## The Mathematical Framework

### 1. Data Standardization
Raw options chains are fetched and standard Black-Scholes inversion (via Brent's root-finding method) is applied to extract market implied volatilities. To ensure the model scales across different asset classes and index levels, the input space is transformed into **Log-Moneyness** ($k$):
$$k = \ln\left(\frac{K}{F_T}\right)$$

### 2. Bayesian Non-Parametric Modeling
A Gaussian Process (GP) is used to model the volatility surface. Unlike standard parametric models (e.g., SABR, SVI), GPs provide rigorous uncertainty quantification in sparse data regions (like deep out-of-the-money options).
* **Kernel Selection:** A **Matérn 5/2 Kernel** is utilized alongside a **White Kernel**. The Matérn kernel captures the realistic financial "roughness" of the surface, while the White kernel absorbs the microstructural noise inherent in bid-ask spreads.

### 3. Arbitrage Detection via Dupire's Formula
A standard GP natively fits the noise, generating a surface riddled with static arbitrage. To prove this, the pipeline extracts the **Local Volatility Surface** using Dupire's Equation:

$$
\sigma_{LV}^2 = \frac{\frac{\partial w}{\partial T}}{1 - \frac{k}{w}\frac{\partial w}{\partial k} + \frac{1}{4}\left(-\frac{1}{4} - \frac{1}{w} + \frac{k^2}{w^2}\right)\left(\frac{\partial w}{\partial k}\right)^2 + \frac{1}{2}\frac{\partial^2 w}{\partial k^2}}
$$

Numerical partial derivatives are computed across the GP grid to test for:
1. **Calendar Arbitrage:** Monotonicity in time ($\frac{\partial w}{\partial T} \ge 0$).
2. **Butterfly Arbitrage:** Strict convexity in log-moneyness (probability density must be positive).

*Result: The unconstrained GP exhibited a ~44% failure rate, yielding imaginary local volatility.*

### 4. Arbitrage-Free Projection
To correct the violations, the flawed GP surface is treated as a Bayesian prior. A constrained optimization and sequential smoothing algorithm projects the surface onto the nearest arbitrage-free subspace, enforcing strict time-monotonicity and strike-convexity.

### The Baseline Surface vs. Arbitrage Detection
The baseline GP fits the data perfectly but creates localized areas of negative variance.
### The Cleaned Local Volatility Surface
After projection, the surface adheres to the laws of quantitative finance and is ready for exotic derivative pricing.
## Cross-Asset Architecture

The `VolatilitySurfaceEngine` class is designed to adapt to varying market liquidity dynamically:
* **SPY (S&P 500):** Dense data; the GP confidently maps the institutional volatility smirk.
* **QQQ (Nasdaq 100):** Captures higher noise variance reflecting wider intraday tech bid-ask spreads.
* **IWM (Russell 2000):** Adapts kernel length scales aggressively to handle severe options sparseness, relying on uncertainty quantification to prevent surface collapse.
