# A Tale of Two Grids: Forecasting Electricity in Texas vs. New York

**Lucas Azenha & Anya Tu — Columbia University, 2026**

Can a seasonal ARIMA model, paired with a GARCH volatility model, forecast daily statewide electricity in Texas and New York? And what does the difference in forecast accuracy say about the two grids?

## Approach

1. **Data:** daily electricity data for ERCOT (Texas) and NYISO (New York) from the [U.S. Energy Information Administration API](https://www.eia.gov/opendata/).
2. **Mean model:** SARIMA `(1,0,2)×(0,1,1)₇` on the log series, with weekly seasonality and an 80/20 train/test split. Stationarity is checked with an ADF test and the order with `auto_arima`.
3. **Volatility model:** GARCH(1,1) with skewed Student's t errors on the SARIMA residuals, giving a 95% volatility band around the forecast.
4. **Diagnostics:** Ljung-Box, Jarque-Bera, and ARCH-LM tests on the residuals.
5. **Extension:** cooling and heating degree days (from [Open-Meteo](https://open-meteo.com/)) as external inputs for New York.

## Results

| Series | Model | Window | Test MAPE |
|---|---|---|---|
| Texas | SARIMA-GARCH | Oct 13, 2025 – Jun 20, 2026 (251 days) | 13.58% |
| New York | SARIMA-GARCH | Apr 28 – Jun 29, 2026 (63 days) | 5.82% |
| New York | SARIMAX-GARCH with weather | Apr 28 – Jun 29, 2026 (63 days) | 6.28% |

New York is far easier to forecast than Texas. Texas residuals are strongly heavy-tailed: the model misses occasional large shocks such as Winter Storm Fern (January 2026), which a linear SARIMA structure cannot anticipate. The GARCH layer captures those shocks as time-varying volatility.

## Run it

```bash
pip install -r requirements.txt
export EIA_API_KEY=your_key_here   # free key: https://www.eia.gov/opendata/register.php
jupyter notebook tale_of_two_grids.ipynb
```

The notebook also runs in Google Colab. It asks for the key if `EIA_API_KEY` is not set.

## Limitations and next steps

- The analysis windows are short because the API returns at most 5,000 rows per request. Paging through the full history would allow longer training periods and yearly seasonality.
- The notebook's `APPLY_FIXES` switch corrects three data-handling issues in the original analysis (described in the notebook). The results above are from the original settings.
- The weather model uses observed weather in the test period, so its error is optimistic compared with a real forecast.
- Next: rolling-origin evaluation and machine learning baselines (gradient boosting, LSTM).
