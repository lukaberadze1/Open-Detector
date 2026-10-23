import pytest

from opendetector import DetectorConfig


def test_slim_profile_is_desklib_float32_on_cpu():
    slim = DetectorConfig.for_profile("slim")
    assert slim.desklib_model and slim.glyph_model is None and slim.binoculars_observer is None
    assert slim.device == "cpu" and slim.dtype == "float32"
    assert slim.combiner_path.name == "combiner_slim.json"


def test_full_profile_is_the_default_and_unknown_profiles_fail():
    assert DetectorConfig.for_profile("full") == DetectorConfig()
    with pytest.raises(ValueError, match="unknown profile"):
        DetectorConfig.for_profile("tiny")
