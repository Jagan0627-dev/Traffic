# Traffic Volume Forecasting Using Machine Learning

A local Streamlit GUI for estimating hourly traffic volume from user-entered date,
time, and weather conditions. It trains a Random Forest on the included Metro
Interstate Traffic Volume dataset (or a replacement CSV uploaded by the user) and
reports chronological holdout metrics.

The dataset represents a highway in Minnesota, USA. This application is not a live
traffic monitoring system and does not claim to use a sequence-based forecast.

## Project structure

```text
Marrix/
├── app.py
├── forecasting.py
├── requirements.txt
├── README.md
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
└── tests/
    └── test_forecasting.py
```

## Dataset setup

The app includes the Metro Interstate CSV in `data/` and loads it by default. To
replace it, download the [Metro Interstate Traffic Volume Dataset from UCI](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume),
extract the CSV, then upload it using the optional sidebar file uploader.

The app expects the original dataset fields: `date_time`, `traffic_volume`, `temp`,
`rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, and `holiday`. The full dataset is
included. The model trains on the selected dataset before enabling prediction.
Dataset citation and license: Metro Interstate Traffic Volume dataset, UCI Machine
Learning Repository, https://doi.org/10.24432/C5X60B; licensed under CC BY 4.0.

## Run locally in VS Code

Open this folder in VS Code and run these commands in the integrated PowerShell
terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, run the commands using the virtual
environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in the browser at the local Streamlit URL printed in the terminal.

## Example input defaults

- Date: `2018-09-30`
- Hour: `17`
- Temperature: `14.85 °C` (equivalent to the dataset's `288 K` example)
- Rainfall: `0 mm/h`
- Snowfall: `0 mm/h`
- Cloud cover: `40%`
- Weather: `Rain` (when present in the uploaded data)
- Holiday: `None` (when present in the uploaded data)

Temperature is entered in Celsius and converted to Kelvin before prediction because
the dataset stores temperature in Kelvin. Weather and holiday dropdown options come
from the selected dataset. Numeric inputs and required columns are validated; missing
feature values are imputed by the fitted Scikit-learn pipeline.

## Data-science lifecycle

1. **Business understanding:** estimate hourly traffic volume to support planning.
2. **Data understanding:** review the upload, record count, time coverage, and sample
   rows.
3. **Data preparation:** remove duplicate rows, discard invalid timestamps/targets,
   parse numeric features, and derive calendar and cyclic time features.
4. **Modeling:** train a Scikit-learn preprocessing pipeline with median/mode
   imputation, one-hot encoding for weather and holiday, and a Random Forest
   Regressor.
5. **Evaluation:** use the final 20% of time-ordered records as holdout data; display
   MAE, MSE, RMSE, R², and an actual-versus-predicted chart.
6. **Use and monitoring:** predict a selected hour with the trained pipeline and
   repeat evaluation as newer labeled records become available.

The input date can be any date, but dates after the model's observed training period
are outside the training coverage. Predictions are estimates and may not reflect
changes in traffic patterns or conditions not represented by the dataset.

## Run tests

```powershell
python -m unittest discover -s tests -v
```
