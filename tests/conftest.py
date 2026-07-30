import importlib.util
import os
import sys
import types

import pytest

DEFAULT_RP_PATH = r"F:\magic\extensions\sd-webui-regional-prompter"


@pytest.fixture(scope="session")
def rp_regions():
    """导入 RP 自己的 regions 模块作为布局参照。

    RP 的 regions.py 会 `from modules import devices`，该模块只在 WebUI 运行时存在，
    因此这里注入一个 stub。若 RP 源码不可用则 skip，而不是 fail。
    """
    root = os.environ.get("RP_REFERENCE_PATH", DEFAULT_RP_PATH)
    path = os.path.join(root, "scripts", "regions.py")
    if not os.path.isfile(path):
        pytest.skip(f"RP 参考源码不存在：{path}（设置 RP_REFERENCE_PATH 以启用）")

    if "modules" not in sys.modules:
        modules_stub = types.ModuleType("modules")
        devices_stub = types.ModuleType("modules.devices")
        devices_stub.device = "cpu"
        modules_stub.devices = devices_stub
        sys.modules["modules"] = modules_stub
        sys.modules["modules.devices"] = devices_stub

    spec = importlib.util.spec_from_file_location("rp_regions_reference", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"RP 参考源码导入失败：{exc}")
    return module
