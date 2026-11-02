import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "online_retail_eda.ipynb"


def test_eda_notebook_is_valid_and_calls_reusable_pipeline():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"][0]["cell_type"] == "markdown"
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "preprocess_transactions" in code
    assert "plot_eda" in code
    assert "download_dataset.py" in code
