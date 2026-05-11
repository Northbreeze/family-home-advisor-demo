# Static Client Demo Deployment

This deployment mode lets the client explore the packaged workbook without running Realtor.ca scraping in the cloud.

## Files Required

- `app.py`
- `buyer_profile.py`
- `data_cleaning.py`
- `scoring.py`
- `family_home_advisor_client_report.xlsx`
- `requirements-demo.txt`

Optional but okay to include:

- `README.md`
- `.streamlit/config.toml`

Do not rely on the cloud app for refreshing listings. Refresh locally, rebuild `family_home_advisor_client_report.xlsx`, then redeploy or push the updated workbook.

## Streamlit Community Cloud

1. Create a GitHub repo, for example `family-home-advisor-demo`.
2. Upload the required files above.
3. In Streamlit Community Cloud, create a new app from the repo.
4. Main file path: `app.py`.
5. Advanced settings / secrets:

```toml
APP_MODE = "demo"
```

6. If Streamlit Cloud does not automatically use `requirements-demo.txt`, rename it to `requirements.txt` in the demo repo.

## Local Demo Mode

PowerShell:

```powershell
$env:APP_MODE="demo"
& "C:\Users\NASTABA\.conda\envs\ml_env\python.exe" -m streamlit run app.py
```

In demo mode:

- the refresh button is hidden;
- the app reads `family_home_advisor_client_report.xlsx`;
- the client can change scoring sliders and presets;
- the client can download the currently filtered/ranked Excel.
