# Family Home Advisor

A small Streamlit MVP for turning the existing Realtor.ca housing-analysis exports into a client-facing recommendation app.

The notebook pipeline is still intact. The app reads the latest enriched Excel output, cleans stale open-house dates, applies transparent buyer-profile scoring, shows listings on a map, and exports ranked recommendations.

## Default Input

The app looks for files in this order:

1. `family_home_advisor_client_report.xlsx`
2. `houses_client_ready_school_scores.xlsx`
3. `houses_with_school_catchments_latest_scores.xlsx`
4. Latest `North_West_Vancouver_Houses_Open_Houses_WORKING_URLS_*.xlsx`

If columns are missing, the app shows a Streamlit warning and uses safe defaults instead of crashing.

## Setup

Use your conda environment:

```bash
/c/Users/NASTABA/.conda/envs/ml_env/python -m pip install -r requirements.txt
```

On PowerShell:

```powershell
& "C:\Users\NASTABA\.conda\envs\ml_env\python.exe" -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Or from PowerShell with the conda environment:

```powershell
& "C:\Users\NASTABA\.conda\envs\ml_env\python.exe" -m streamlit run app.py
```

## Refresh Listings

The Streamlit sidebar includes a `Refresh listings now` button. It runs `refresh_pipeline.py`, which:

- fetches current North Vancouver and West Vancouver listings with `pyRealtor`;
- keeps house listings under the configured max price;
- optionally restricts to listings with open houses;
- enriches listings with local school catchments and Fraser scores;
- adds approximate major-road/noise risk fields;
- writes a fresh `family_home_advisor_client_report.xlsx`.

You can also run the refresh from PowerShell:

```powershell
& "C:\Users\NASTABA\.conda\envs\ml_env\python.exe" -X utf8 refresh_pipeline.py
```

To test enrichment without fetching Realtor.ca again:

```powershell
& "C:\Users\NASTABA\.conda\envs\ml_env\python.exe" -X utf8 refresh_pipeline.py --use-existing-listing-file
```

BC Assessment land/building values are not bulk-fetched by default. The refreshed workbook includes assessment placeholder/status columns and a `BC Assessment Search Link` for manual verification.

## MVP Features

- Loads the newest enriched listing workbook.
- Parses open-house strings such as `May 02/26 - 2:00 PM To 4:00 PM`.
- Creates `open_house_raw`, `next_open_house`, `last_open_house`, and `open_house_status`.
- Adds sidebar controls for price, bedrooms, Fraser score, buyer priorities, city, and high-noise exclusion.
- Includes three preset profiles: `Quiet Family Profile`, `Top School Profile`, and `Value Buyer Profile`.
- Uses deterministic scoring rather than LLM scoring.
- Applies open-house bonus only to upcoming open houses.
- Uses existing noise fields when available and otherwise marks noise risk as estimated.
- Shows an interactive Folium map with match/noise marker colors and listing popups.
- Shows a ranked recommendation table.
- Exports Excel with `Client Preferences`, `Top Recommendations`, `All Filtered Listings`, `Excluded Homes`, and `Scoring Method` sheets.
- Includes a rule-based buyer profile helper that works without an OpenAI API key.

## Code Structure

- `app.py`: Streamlit UI and map rendering.
- `data_cleaning.py`: workbook selection, column normalization, open-house parsing, Google Maps coordinate extraction, and noise-risk enrichment.
- `buyer_profile.py`: preset buyer profiles and rule-based buyer text parsing.
- `scoring.py`: deterministic scoring, filtering, table columns, marker colors, and Excel export.
