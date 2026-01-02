
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import nbformat
from dagster import op, job, Field, StringSource, Int, Definitions, get_dagster_logger


_JUPYTER_MAGIC_PREFIXES = ("!", "%", "%%")

def _sanitize_cell_source(source: str) -> str:
    """
    Remove/skip notebook-only lines (shell escapes, magics) while keeping Python code intact.
    We keep pure Python `...` (Ellipsis) lines, as they are valid Python.
    """
    out_lines: List[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(_JUPYTER_MAGIC_PREFIXES):
            # Skip lines like: !pip install ..., %matplotlib inline, %%time, etc.
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()

@op(
    config_schema={
        "notebook_path": Field(StringSource, description="Path to the .ipynb notebook to execute."),
        "working_dir": Field(StringSource, default_value=str(Path.cwd()), description="Working directory during execution."),
        "stop_after_cell": Field(Int, default_value=-1, description="Execute up to this 0-based code-cell index (inclusive). -1 runs all."),
    }
)
def run_notebook(context) -> Dict[str, Any]:
    """
    Executes the notebook's code cells sequentially (like 'Run All'),
    skipping notebook magics/shell commands.

    Returns a dict with basic execution metadata.
    """
    cfg = context.op_config
    notebook_path = Path(os.path.expanduser(str(cfg["notebook_path"]))).resolve()
    working_dir = Path(os.path.expanduser(str(cfg["working_dir"]))).resolve()
    stop_after_cell = int(cfg.get("stop_after_cell", -1))

    logger = get_dagster_logger()
    logger.info(f"Loading notebook: {notebook_path}")
    nb = nbformat.read(str(notebook_path), as_version=4)

    # Execution namespace
    ns: Dict[str, Any] = {
        "__name__": "__dagster_notebook_exec__",
        # Provide a helpful variable often used in notebooks
        "DAGSTER_CONTEXT": context,
    }

    # Run inside the chosen working directory
    working_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(str(working_dir))

    executed_cells = 0
    skipped_cells = 0
    try:
        code_cell_index = -1
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            code_cell_index += 1
            if stop_after_cell >= 0 and code_cell_index > stop_after_cell:
                break

            src = cell.source or ""
            sanitized = _sanitize_cell_source(src)

            if not sanitized:
                skipped_cells += 1
                continue

            logger.info(f"Executing code cell #{code_cell_index} (len={len(sanitized)} chars)")
            try:
                exec(compile(sanitized, filename=f"{notebook_path.name}:cell_{code_cell_index}", mode="exec"), ns, ns)
                executed_cells += 1
            except Exception as e:
                logger.exception(f"Error executing cell #{code_cell_index}")
                raise

    finally:
        os.chdir(str(old_cwd))

    # Collect a light summary (avoid serializing big dataframes)
    created_keys = sorted([k for k in ns.keys() if not k.startswith("_") and k not in ("os", "re", "Path", "nbformat")])
    return {
        "notebook_path": str(notebook_path),
        "working_dir": str(working_dir),
        "executed_cells": executed_cells,
        "skipped_cells": skipped_cells,
        "namespace_keys_sample": created_keys[:50],
        "namespace_keys_count": len(created_keys),
    }

@job
def notebook_job():
    run_notebook()

def get_definitions(default_notebook_path: Optional[str] = None) -> Definitions:
    return Definitions(
        jobs=[notebook_job]
    )
