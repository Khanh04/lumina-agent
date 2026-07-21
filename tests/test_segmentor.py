import runpy


def test_segmentor_selfcheck():
    # Exercises the OpenCV mask routes (global/sky/radial) without loading MediaPipe.
    runpy.run_module("app.cv.segmentor", run_name="__main__")
