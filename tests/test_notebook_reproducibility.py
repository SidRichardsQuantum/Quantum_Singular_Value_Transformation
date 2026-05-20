import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_random_generators_use_visible_default_seed():
    """
    Keep notebook randomness reproducible and easy to audit.
    """
    notebooks = [
        *sorted((REPO_ROOT / "notebooks" / "tutorials").glob("*.ipynb")),
        *sorted((REPO_ROOT / "notebooks" / "benchmarks").glob("*.ipynb")),
        *sorted((REPO_ROOT / "notebooks" / "real_examples").glob("*.ipynb")),
    ]
    offenders = []

    for path in notebooks:
        raw_text = path.read_text(encoding="utf-8")
        if "default_rng" not in raw_text and "np.random" not in raw_text:
            continue

        notebook = json.loads(raw_text)
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        if "default_rng" not in source and "np.random" not in source:
            continue
        if "SEED = 0" not in source:
            offenders.append(f"{path.relative_to(REPO_ROOT)} missing SEED = 0")
        if re.search(r"default_rng\(\s*\d+", source):
            offenders.append(
                f"{path.relative_to(REPO_ROOT)} uses an inline RNG seed",
            )

    assert offenders == []
