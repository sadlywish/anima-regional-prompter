import os
import sys

import pytest

import lib_anima_rp


def test_package_importable():
    assert lib_anima_rp.__doc__


def test_rp_reference_available(rp_regions):
    """验证 RP 参考源码可用且包含所需属性"""
    assert hasattr(rp_regions, "split_l2")
    assert hasattr(rp_regions, "ratiosdealer")


def test_sys_modules_cleanup_after_fixture(rp_regions):
    """验证 rp_regions fixture 用完后，sys.modules 中的 stub 被清理"""
    # 如果 fixture 成功执行（不是 skip），则不应该在 sys.modules 中残留 stub
    # 但要排除真实模块已存在的情况
    assert "modules" not in sys.modules or isinstance(
        sys.modules.get("modules"), type(rp_regions)
    ) is False, "stub 'modules' 未被清理"


def test_explicit_rp_reference_path_missing_fails(monkeypatch):
    """验证：RP_REFERENCE_PATH 显式设置为不存在的路径时，应该 FAIL 而不是 skip"""
    import importlib.util
    import sys

    # 直接引入 conftest 中的函数
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")
    spec = importlib.util.spec_from_file_location("conftest_test", conftest_path)
    conftest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest_mod)

    with pytest.raises(FileNotFoundError, match="RP_REFERENCE_PATH 被显式设置"):
        conftest_mod._validate_rp_path(
            r"F:\definitely\does\not\exist", rp_path_explicit=True
        )
