import importlib.util
import os
import sys
import types

import pytest

DEFAULT_RP_PATH = r"F:\magic\extensions\sd-webui-regional-prompter"


def _validate_rp_path(root, rp_path_explicit):
    """验证 RP 路径是否存在，根据是否显式设置决定 skip 或 fail。

    返回 regions.py 的完整路径。

    参数：
        root: RP 根路径
        rp_path_explicit: 是否通过环境变量显式设置了路径
    """
    path = os.path.join(root, "scripts", "regions.py")
    if not os.path.isfile(path):
        if rp_path_explicit:
            # RP_REFERENCE_PATH 被显式设置但路径不存在 → FAIL
            raise FileNotFoundError(
                f"RP 参考源码不存在：{path}（RP_REFERENCE_PATH 被显式设置）"
            )
        else:
            # 默认路径不存在 → skip（允许别人 clone 仓库也能跑）
            pytest.skip(f"RP 参考源码不存在：{path}（设置 RP_REFERENCE_PATH 以启用）")
    return path


def _load_rp_regions(path, rp_path_explicit):
    """加载 RP 的 regions 模块，管理 stub 的生命周期。

    Stub 仅在 exec_module 期间存在，之后立刻移除。

    参数：
        path: regions.py 的完整路径
        rp_path_explicit: 是否通过环境变量显式设置了路径

    返回：加载的 regions 模块
    """
    # 记录要注入的 stub 键（仅在不存在真实模块时注入）
    stubs_to_inject = []

    try:
        # 检查并注入 stub
        if "modules" not in sys.modules:
            modules_stub = types.ModuleType("modules")
            devices_stub = types.ModuleType("modules.devices")
            devices_stub.device = "cpu"
            # 给 stub 打上私有标记属性，以便测试验证清理
            modules_stub._anima_rp_stub = True
            devices_stub._anima_rp_stub = True
            modules_stub.devices = devices_stub
            sys.modules["modules"] = modules_stub
            sys.modules["modules.devices"] = devices_stub
            stubs_to_inject = ["modules", "modules.devices"]

        # 导入模块
        spec = importlib.util.spec_from_file_location("rp_regions_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module
    finally:
        # 清理 stub（仅清理自己注入的）
        for key in stubs_to_inject:
            if key in sys.modules:
                del sys.modules[key]


@pytest.fixture(scope="session")
def rp_regions():
    """导入 RP 自己的 regions 模块作为布局参照。

    RP 的 regions.py 会 `from modules import devices`，该模块只在 WebUI 运行时存在，
    因此这里在加载时注入一个 stub，导入完成后立刻移除，避免污染真实的 modules 模块。

    行为：
    - RP_REFERENCE_PATH 未设置且默认路径不存在 → skip（允许别人 clone 仓库跑测试）
    - RP_REFERENCE_PATH 被显式设置 → 路径不存在或导入失败都 FAIL
    - 默认路径存在但导入失败 → FAIL
    """
    rp_path_explicit = "RP_REFERENCE_PATH" in os.environ
    root = os.environ.get("RP_REFERENCE_PATH", DEFAULT_RP_PATH)

    path = _validate_rp_path(root, rp_path_explicit)
    return _load_rp_regions(path, rp_path_explicit)
