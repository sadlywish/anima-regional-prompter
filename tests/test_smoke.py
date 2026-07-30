import lib_anima_rp


def test_package_importable():
    assert lib_anima_rp.__doc__


def test_rp_reference_available(rp_regions):
    assert hasattr(rp_regions, "split_l2")
    assert hasattr(rp_regions, "ratiosdealer")
