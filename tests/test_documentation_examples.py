import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _readme_quick_example_blocks() -> list[str]:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Quick Example", maxsplit=1)[1].split(
        "## Package Map", maxsplit=1
    )[0]
    return re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)


@pytest.mark.parametrize("source", _readme_quick_example_blocks())
def test_readme_quick_examples_execute_as_standalone_snippets(source):
    namespace = {"__name__": "__documentation_example__"}

    exec(compile(source, "README.md", "exec"), namespace)
