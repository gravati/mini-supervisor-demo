
Running tests
---------------
This project doesn't include pytest by default. A small smoke test for the text adapter is in `tests/test_text_adapter.py`.

To run the smoke test without installing pytest, run the helper script:

```cmd
python tmp_run_test_no_pytest.py
```

If you'd like to run with pytest, install it first:

```cmd
pip install pytest
python -m pytest -q
```

