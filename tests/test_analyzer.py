import runpy


def test_analyzer_selfcheck():
    # Global stats + masked-mean telemetry against a stub segmentor (no MediaPipe).
    runpy.run_module("app.cv.analyzer", run_name="__main__")
