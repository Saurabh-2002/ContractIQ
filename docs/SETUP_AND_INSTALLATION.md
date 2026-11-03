# Setup and Installation (Windows)

All commands are for **Windows PowerShell**.

> These steps were written for Windows, but they could not be run during development. If a step fails, check [Troubleshooting](#11-troubleshooting).

## 1. Python installation requirements

Install **64-bit Python 3.12** from [python.org/downloads](https://www.python.org/downloads/). On the first installer screen, tick **Add python.exe to PATH**.

Python 3.12 is recommended because the machine-learning packages are widely available for it. ContractIQ needs at least Python 3.10.

Check the installation:

```powershell
py -3.12 --version
```

You also need about 3 GB of free disk space for dependencies and the embedding model, plus internet access for installation and the first run.

## 2. VS Code setup

1. Install [VS Code](https://code.visualstudio.com/).
2. Open the Extensions panel with `Ctrl+Shift+X`.
3. Install the **Python** extension from Microsoft.

## 3. ZIP extraction

1. Right-click `ContractIQ_MVP.zip` and choose **Extract All**.
2. Extract to a short path, such as `C:\Projects`.

This creates `C:\Projects\ContractIQ`. A short path helps avoid Windows path-length errors.

## 4. Opening the project in VS Code

1. Open VS Code.
2. Select **File > Open Folder**.
3. Choose the `ContractIQ` folder, which contains `app.py`.
4. Open a terminal with **Terminal > New Terminal**. Make sure that it is PowerShell.

## 5. Creating a virtual environment

```powershell
py -3.12 -m venv .venv
```

## 6. Activating the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

If PowerShell blocks the script, run this and then activate again. It applies only to the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 7. Installing requirements

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This can take several minutes, because PyTorch is a large download.

## 8. Creating the .env file

```powershell
Copy-Item .env.example .env
notepad .env
```

## 9. Adding the Gemini API key

1. Create a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. In Notepad, replace the placeholder so that the line reads as follows, using your own key:

   ```text
   GEMINI_API_KEY=your_actual_key
   ```

3. Save and close.

Never share this key or commit `.env`. See [API_CONFIGURATION.md](API_CONFIGURATION.md).

## 10. Verifying and running the application

Check the real stack first. This downloads the embedding model on the first run and writes the results to `docs\LOCAL_VERIFICATION_RESULTS.md`:

```powershell
python verify_setup.py
```

Run the automated tests:

```powershell
pytest
pytest -m integration
```

Start the app:

```powershell
streamlit run app.py
```

Your browser should open at `http://localhost:8501`. To stop the app, press `Ctrl+C` in the terminal.

To run it again later:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## 11. Troubleshooting

| Problem | Solution |
|---|---|
| `py` is not recognised | Reinstall Python with **Add python.exe to PATH**, then open a new terminal |
| `Activate.ps1 cannot be loaded` | Run the `Set-ExecutionPolicy` command in step 6 |
| `pip` errors with long paths | Move the project to a short path, such as `C:\Projects\ContractIQ` |
| No matching distribution for `torch` or `faiss-cpu` | Check that you are using 64-bit Python 3.12 with `python --version` |
| A package version cannot be resolved | Run `pip install` for that package without a version, then record the installed version with `pip freeze` |
| `DLL load failed` or `WinError 1114` when loading PyTorch | Install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist), then restart the terminal |
| `OMP: Error #15` | Start the app with `streamlit run app.py` so that `config/settings.py` applies the fix. If it persists, run `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first |
| The embedding model fails to download | Check your connection, firewall or proxy, then try again |
| The app says that no API key was found | Confirm that the file is named `.env`, not `.env.txt`, then restart the app |
| Gemini rejected the API key | Check the key in `.env` for extra spaces or quotes |
| The Gemini model was not found | Set a current model in `.env`. See [API_CONFIGURATION.md](API_CONFIGURATION.md) |
| Port 8501 is in use | Run `streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError` | Activate the virtual environment and run the install command again |
