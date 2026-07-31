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
    """验证 rp_regions fixture 用完后，sys.modules 中的 stub 被清理

    基于身份标记而非类型检查，以便在 Task 10 引入真实 modules 包后仍然正确判定。
    """
    # 验证我们关心的模块键没有残留 stub 标记
    # 即使真实的 modules 包存在，标记也能区分 stub 和真实包
    for key in ["modules", "modules.devices"]:
        if key in sys.modules:
            module = sys.modules[key]
            if hasattr(module, "_anima_rp_stub"):
                pytest.fail(f"stub '{key}' 未被清理（仍有 _anima_rp_stub 标记）")


def test_explicit_rp_reference_path_missing_fails():
    """验证：RP_REFERENCE_PATH 显式设置为不存在的路径时，应该 FAIL 而不是 skip"""
    import importlib.util

    # 直接引入 conftest 中的函数
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")
    spec = importlib.util.spec_from_file_location("conftest_test", conftest_path)
    conftest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest_mod)

    with pytest.raises(FileNotFoundError, match="RP_REFERENCE_PATH 被显式设置"):
        conftest_mod._validate_rp_path(
            r"F:\definitely\does\not\exist", rp_path_explicit=True
        )


def test_sys_modules_cleanup_with_real_module_present():
    """验证：当真实 modules 包已存在时，清理仍然正确（不污染真实包）

    这是为 Task 10 场景设计的测试。Task 10 接线 Forge 后，真实的 modules 包
    会存在于 sys.modules 中。此时 stub 注入应该被跳过（因为真实包已存在），
    且测试验证逻辑应该仍然能正确判定没有 stub 残留。
    """
    import importlib.util
    import types

    # 直接引入 conftest 中的函数
    conftest_path = os.path.join(os.path.dirname(__file__), "conftest.py")
    spec = importlib.util.spec_from_file_location("conftest_test", conftest_path)
    conftest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest_mod)

    # 构造一个假的真实 modules 包（不带 _anima_rp_stub 标记）
    real_modules = types.ModuleType("modules")
    real_devices = types.ModuleType("modules.devices")
    real_devices.device = "cuda"  # 用不同值区分
    real_modules.devices = real_devices

    # 临时将真实包注入
    old_modules = sys.modules.get("modules")
    old_devices = sys.modules.get("modules.devices")
    try:
        sys.modules["modules"] = real_modules
        sys.modules["modules.devices"] = real_devices

        # 获取 RP 源码路径（与 rp_regions fixture 相同）
        rp_path = os.path.join(
            os.environ.get(
                "RP_REFERENCE_PATH",
                r"F:\magic\extensions\sd-webui-regional-prompter",
            ),
            "scripts",
            "regions.py",
        )

        # 如果 RP 源码存在，调用 _load_rp_regions
        if os.path.isfile(rp_path):
            module = conftest_mod._load_rp_regions(rp_path, False)
            assert module is not None

            # 验证真实包仍然存在且未被修改
            assert sys.modules.get("modules") is real_modules
            assert sys.modules.get("modules.devices") is real_devices
            assert sys.modules["modules"].devices.device == "cuda"

            # 验证没有 stub 残留（关键模块键不存在 _anima_rp_stub 标记）
            for key in ["modules", "modules.devices"]:
                if key in sys.modules:
                    mod = sys.modules[key]
                    if hasattr(mod, "_anima_rp_stub"):
                        pytest.fail(f"stub '{key}' 未被清理（仍有 _anima_rp_stub 标记）")
        else:
            pytest.skip(f"RP 源码不存在：{rp_path}（设置 RP_REFERENCE_PATH 以启用）")
    finally:
        # 恢复原始状态
        if old_modules is not None:
            sys.modules["modules"] = old_modules
        else:
            sys.modules.pop("modules", None)
        if old_devices is not None:
            sys.modules["modules.devices"] = old_devices
        else:
            sys.modules.pop("modules.devices", None)
