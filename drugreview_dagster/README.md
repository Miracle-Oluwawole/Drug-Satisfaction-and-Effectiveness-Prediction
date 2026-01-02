# Drug Review Dagster Pipeline (Notebook Runner)

This Dagster project **executes your uploaded Jupyter notebook as a Dagster job**, preserving the notebook's
cell-by-cell execution order ("Run All") as closely as possible.

## What "exactly converted" means here
- All **Python code cells** are executed sequentially.
- Notebook-only lines are **skipped**:
  - Shell escapes like `!pip install ...`
  - Jupyter magics like `%matplotlib inline`, `%%time`, etc.

> If your notebook contains hard-coded local file paths (e.g., `C:/Users/...`) you will need to either:
> 1) make those files available in the `working_dir`, or  
> 2) edit the notebook to use relative paths / environment variables.

## Files
- `drugreview_dagster/notebook_runner.py` – the Dagster op that loads & executes the notebook
- `drugreview_dagster/definitions.py` – Dagster `defs`
- `requirements.txt` – dependencies seen in the notebook

## Run
From this folder:

```bash
pip install -r requirements.txt
dagster dev -m drugreview_dagster.definitions
```

Then launch the job **notebook_job**.

### Configure the run
In Dagster UI, set op config for `run_notebook`, for example:

```yaml
ops:
  run_notebook:
    config:
      notebook_path: "/mnt/data/x24289337.ipynb"
      working_dir: "/mnt/data/drugreview_dagster"
      stop_after_cell: -1
```

