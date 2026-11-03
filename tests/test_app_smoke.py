"""Runs the real Streamlit app headlessly. Use: pytest -m integration"""

from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Dashboard", "Document Library", "Contract Intelligence", "AI Assistant"]


@pytest.mark.integration
@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=60).run()
    app.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title[0].value in ("ContractIQ", page)
