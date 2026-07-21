import runpy


def test_engine_selfcheck():
    # Runs the assert-based self-check embedded in app/cv/engine.py (blend math + allowlist).
    runpy.run_module("app.cv.engine", run_name="__main__")
