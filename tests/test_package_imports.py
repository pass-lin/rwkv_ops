"""rwkv_ops 包完整性测试：包内 import 一致性与公共 API 导入冒烟。

静态检查直接解析源码 AST，不 import 任何后端；运行期检查穷举
(KERAS_BACKEND, KERNEL_TYPE) 全组合，每个组合都在子进程里执行
`import rwkv_ops`，避免把某个 Keras 后端锁定进当前 pytest 进程
（与 tests/conftest.py 的约定一致）。

新增算子无需改动本文件的组合矩阵：`__init__.py` 里暴露的符号会被自动纳入
检查；加速桥接模块按 `*_jax_triton*` / `*_torch_triton*` 命名即自动纳入。

设置环境变量 RWKV_OPS_IMPORT_CHECK_ROOT 可把被测包目录指向其它副本
（例如已发布 wheel 解包后的目录），用于复核发布物。
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_PACKAGE_NAME = "rwkv_ops"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_PACKAGE_ROOT = _PROJECT_ROOT / _PACKAGE_NAME

# AGENTS.md §1.1 列出的公共 API：算子实例 + 工厂函数。
_PUBLIC_API = (
    "generalized_delta_rule",
    "generalized_delta_rule_inference",
    "rwkv7_op",
    "rwkv7_op_inference",
    "rnn_generalized_delta_rule",
    "rwkv7_op_rnn",
    "generalized_delta_rule_sane",
    "generalized_delta_rule_sane_inference",
    "rwkv7_op_sane",
    "rwkv7_op_sane_inference",
    "rnn_generalized_delta_rule_sane",
    "rwkv7_op_sane_rnn",
    "rwkv6_op",
    "RWKV6_OP",
    "gated_delta_net_recurrent",
    "gated_delta_net_recurrent_inference",
    "gated_delta_net_recurrent_single_step",
    "gated_delta_net_recurrent_sane",
    "gated_delta_net_recurrent_sane_inference",
    "gated_delta_net_recurrent_sane_single_step",
    "gated_delta_net_chunk",
    "gated_delta_net_chunk_sane",
    "delta_net_recurrent",
    "delta_net_recurrent_inference",
    "delta_net_recurrent_single_step",
    "delta_net_recurrent_sane",
    "delta_net_recurrent_sane_inference",
    "delta_net_recurrent_sane_single_step",
    "delta_net_chunk",
    "delta_net_chunk_sane",
    "mhc_pre_op",
    "mhc_post_op",
    "get_generalized_delta_rule",
    "get_rnn_generalized_delta_rule",
    "get_generalized_delta_rule_sane",
    "get_rnn_generalized_delta_rule_sane",
    "get_rwkv6_kernel",
    "get_mhc_kernel",
    "get_gated_delta_net_recurrent",
    "get_gated_delta_net_recurrent_inference",
    "get_gated_delta_net_recurrent_single_step",
    "get_gated_delta_net_chunk",
    "get_gated_delta_net_chunk_sane",
    "get_gated_delta_net_recurrent_sane",
    "get_gated_delta_net_recurrent_sane_inference",
    "get_gated_delta_net_recurrent_sane_single_step",
    "get_delta_net_recurrent",
    "get_delta_net_recurrent_inference",
    "get_delta_net_recurrent_single_step",
    "get_delta_net_chunk",
    "get_delta_net_chunk_sane",
    "get_delta_net_recurrent_sane",
    "get_delta_net_recurrent_sane_inference",
    "get_delta_net_recurrent_sane_single_step",
)

# 后端 × KERNEL_TYPE 穷举：每个组合都必须能 `import rwkv_ops` 并暴露全部 API。
_BACKENDS = ("torch", "jax", "numpy", "tensorflow", "openvino")
_KERNEL_TYPES = ("native", "triton", "cuda")

_BACKEND_REQUIRES = {
    "torch": ("torch",),
    "jax": ("jax",),
    "numpy": (),
    "tensorflow": ("tensorflow",),
    "openvino": ("openvino",),
}

# GPU 机器上 cuda 组合会在导入期触发 RWKV-6 FFI / C++ 扩展编译，故标记 slow
# （`pytest -m "not slow"` 可跳过）；无 CUDA 机器上 cuda 组合静默回退 native，
# 导入代价与 native 组合相同。
_IMPORT_CASES = tuple(
    pytest.param(
        backend,
        kernel_type,
        id=f"{backend}-{kernel_type}",
        marks=pytest.mark.slow if kernel_type == "cuda" else (),
    )
    for backend in _BACKENDS
    for kernel_type in _KERNEL_TYPES
)

_IMPORT_SCRIPT = """
import importlib

import rwkv_ops

missing = [name for name in %(exposed)r if not hasattr(rwkv_ops, name)]
if missing:
    raise SystemExit("rwkv_ops 导入后缺失符号: %%s" %% (missing,))

not_callable = [
    name for name in %(documented)r if not callable(getattr(rwkv_ops, name, None))
]
if not_callable:
    raise SystemExit("rwkv_ops 公共 API 不可调用: %%s" %% (not_callable,))

for module_name in %(extra)r:
    importlib.import_module(module_name)

if rwkv_ops.KERNEL_TYPE != %(kernel_type)r:
    raise SystemExit("KERNEL_TYPE 解析错误: %%r" %% (rwkv_ops.KERNEL_TYPE,))

print("rwkv_ops %%s 导入冒烟通过" %% rwkv_ops.__version__)
"""


def _exposed_names(package_root):
    """从 rwkv_ops/__init__.py 推导导入后应存在的公共符号名。

    新增算子只要在 `__init__.py` 中暴露，就自动纳入全部后端 × KERNEL_TYPE 的
    导入矩阵，无需在测试里额外登记。只取无条件绑定的名字：写在 if/try 分支里
    的名字（如默认分支才 `import keras`）不保证存在。

    Args:
        package_root: Path，rwkv_ops 包目录。

    Returns:
        list[str]: 不以 "_" 开头的模块级无条件绑定名字，按名排序。
    """
    init_path = package_root / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    names = _bound_names(tree.body, recurse_branches=False)
    return sorted(name for name in names if not name.startswith("_"))


@pytest.fixture(scope="session")
def package_root():
    """被测 rwkv_ops 包目录。

    默认检查仓库源码；RWKV_OPS_IMPORT_CHECK_ROOT 可指向其它副本。

    Returns:
        Path: rwkv_ops 包目录的绝对路径。
    """
    override = os.environ.get("RWKV_OPS_IMPORT_CHECK_ROOT")
    if override:
        return Path(override).resolve()
    return _SOURCE_PACKAGE_ROOT


def _pattern_names(target):
    """收集赋值/循环目标里绑定的名字。

    Args:
        target: ast 节点，赋值目标或 for 目标。

    Returns:
        set[str]: 绑定的名字集合；`a.b = ...` 这类属性赋值不绑定名字，返回空集。
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names = set()
        for element in target.elts:
            names |= _pattern_names(element)
        return names
    if isinstance(target, ast.Starred):
        return _pattern_names(target.value)
    return set()


def _bound_names(stmts, recurse_branches=True):
    """收集语句块中所有可能绑定的顶层名字。

    覆盖函数/类定义、赋值、import、循环变量；recurse_branches=True 时同时
    收集 if / try / with / while / match 分支内的同名绑定（模块级条件定义的
    名字同样可用）。

    Args:
        stmts: list[ast.stmt]，模块或分支的语句列表。
        recurse_branches: bool，是否深入条件分支收集名字。

    Returns:
        set[str]: 绑定的名字集合；出现 `import *` 时包含 "*"。
    """
    names = set()
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add("*" if alias.name == "*" else (alias.asname or alias.name))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _pattern_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _pattern_names(node.target)
        elif not recurse_branches:
            continue
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            names |= _pattern_names(node.target)
            names |= _bound_names(node.body)
            names |= _bound_names(node.orelse)
        elif isinstance(node, (ast.If, ast.While)):
            names |= _bound_names(node.body)
            names |= _bound_names(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _bound_names(node.body)
            names |= _bound_names(node.orelse)
            names |= _bound_names(node.finalbody)
            for handler in node.handlers:
                if handler.name:
                    names.add(handler.name)
                names |= _bound_names(handler.body)
        elif isinstance(node, ast.With):
            names |= _bound_names(node.body)
        elif isinstance(node, getattr(ast, "Match", ())):
            for case in node.cases:
                names |= _bound_names(case.body)
    return names


def _collect_modules(package_root):
    """扫描包目录，收集每个模块的顶层绑定名字。

    Args:
        package_root: Path，rwkv_ops 包目录。

    Returns:
        dict: {模块名: {"path": Path, "is_package": bool, "tree": ast.Module,
            "names": set[str]}}。
    """
    modules = {}
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_root.parent).with_suffix("")
        parts = list(relative.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules[".".join(parts)] = {
            "path": path,
            "is_package": is_package,
            "tree": tree,
            "names": _bound_names(tree.body),
        }
    return modules


def _resolve_target(module_name, is_package, level, module):
    """把 import 语句解析成包内绝对模块名。

    Args:
        module_name: str，当前模块的绝对名。
        is_package: bool，当前模块是否为包的 __init__。
        level: int，相对导入层级；0 表示绝对导入。
        module: str 或 None，from ... import 里的模块部分。

    Returns:
        str 或 None: 解析后的绝对模块名；越界时返回 None。
    """
    if level == 0:
        return module
    anchor = module_name if is_package else module_name.rpartition(".")[0]
    parts = anchor.split(".") if anchor else []
    up = level - 1
    if up:
        parts = parts[: len(parts) - up]
    if not parts:
        return None
    if module:
        parts = parts + module.split(".")
    return ".".join(parts)


def _package_imports(module_name, is_package, tree):
    """产出指向本包内部的 import 项。

    含函数体内的延迟 import（工厂函数通常在此导入加速实现）。

    Args:
        module_name: str，当前模块的绝对名。
        is_package: bool，当前模块是否为包的 __init__。
        tree: ast.Module，模块语法树。

    Returns:
        list[tuple]: (行号, 目标模块名, [被导入名字])，入口 * 已被过滤。
    """
    items = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _resolve_target(module_name, is_package, node.level, node.module or "")
        if target is None or not target.startswith(_PACKAGE_NAME):
            continue
        names = [alias.name for alias in node.names if alias.name != "*"]
        if names:
            items.append((node.lineno, target, names))
    return items


def _bridge_modules(package_root, backend):
    """发现某后端需要显式导入的加速桥接模块。

    这些模块只在硬件能力探测通过时才会被 `import rwkv_ops` 顺带导入，
    因此需要单独导入以覆盖无 GPU 环境下的名字错误。

    Args:
        package_root: Path，rwkv_ops 包目录。
        backend: str，"torch" 或 "jax"。

    Returns:
        list[str]: 按名排序的模块绝对名。
    """
    marker = "torch_triton" if backend == "torch" else "jax_triton"
    modules = []
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not any(marker in part for part in parts):
            continue
        modules.append(".".join(parts))
    return sorted(set(modules))


def _run_import_subprocess(package_root, backend, kernel_type, extra_modules):
    """在子进程里执行 import 冒烟脚本。

    Args:
        package_root: Path，被测 rwkv_ops 包目录。
        backend: str，KERAS_BACKEND 取值。
        kernel_type: str，KERNEL_TYPE 取值。
        extra_modules: list[str]，额外显式导入的模块。

    Returns:
        subprocess.CompletedProcess: 已完成的子进程结果（text 模式）。
    """
    script = _IMPORT_SCRIPT % {
        "exposed": _exposed_names(package_root),
        "documented": list(_PUBLIC_API),
        "extra": list(extra_modules),
        "kernel_type": kernel_type,
    }
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    env["KERNEL_TYPE"] = kernel_type
    env.pop("KERNEL_BACKEND", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(package_root.parent), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(package_root.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _assert_subprocess_ok(proc, backend, kernel_type):
    """断言子进程导入成功。

    Args:
        proc: subprocess.CompletedProcess，import 冒烟子进程结果。
        backend: str，KERAS_BACKEND 取值，用于错误信息。
        kernel_type: str，KERNEL_TYPE 取值，用于错误信息。

    Raises:
        AssertionError: 子进程返回码非 0。
    """
    assert proc.returncode == 0, (
        f"KERAS_BACKEND={backend}, KERNEL_TYPE={kernel_type} 导入失败"
        f"（returncode={proc.returncode}）\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def _skip_unless_backend_available(backend, extra_requires=()):
    """后端依赖缺失时跳过用例。

    Args:
        backend: str，KERAS_BACKEND 取值。
        extra_requires: tuple[str]，该组合额外需要的模块（如 jax_triton）。

    Raises:
        pytest.skip.Exception: 依赖缺失。
    """
    for module_name in ("keras",) + _BACKEND_REQUIRES[backend] + tuple(extra_requires):
        if importlib.util.find_spec(module_name) is None:
            pytest.skip(f"缺少依赖 {module_name}，跳过 {backend} 后端的导入测试")


@pytest.mark.parametrize("backend,kernel_type", _IMPORT_CASES)
def test_import_rwkv_ops(package_root, backend, kernel_type):
    """各后端/实现类型组合下 `import rwkv_ops` 成功且公共 API 齐全。"""
    _skip_unless_backend_available(backend)
    proc = _run_import_subprocess(package_root, backend, kernel_type, ())
    _assert_subprocess_ok(proc, backend, kernel_type)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_import_accelerator_bridge_modules(package_root, backend):
    """加速桥接模块可独立导入（不受硬件能力探测影响）。"""
    extra = ("triton",) if backend == "torch" else ("jax_triton",)
    _skip_unless_backend_available(backend, extra)
    modules = _bridge_modules(package_root, backend)
    assert modules, f"未发现 {backend} 的加速桥接模块，检查 _bridge_modules 的匹配规则"
    proc = _run_import_subprocess(package_root, backend, "native", modules)
    _assert_subprocess_ok(proc, backend, "native")


def test_intra_package_imports_resolve(package_root):
    """包内 import 的名字必须在目标模块里真实存在。

    这类错误（`__init__.py` 导入了已删除/改名的 kernel 符号）在安装后才会
    在特定后端暴露，因此用静态检查提前拦截。
    """
    modules = _collect_modules(package_root)
    assert _PACKAGE_NAME in modules, f"未找到被测包 {_PACKAGE_NAME}：{package_root}"

    problems = []
    for module_name, info in sorted(modules.items()):
        for lineno, target, imported in _package_imports(
            module_name, info["is_package"], info["tree"]
        ):
            if target not in modules:
                problems.append(f"{module_name}:{lineno} 目标模块 {target} 不存在")
                continue
            available = modules[target]["names"]
            if "*" in available:
                continue
            for name in imported:
                if name in available or f"{target}.{name}" in modules:
                    continue
                relative_target = modules[target]["path"].relative_to(
                    package_root.parent
                )
                problems.append(
                    f"{relative_target}: `{name}` 未定义，"
                    f"但 {info['path'].relative_to(package_root.parent)}:{lineno} "
                    f"从该模块导入"
                )

    assert not problems, "包内 import 引用了不存在的名字：\n" + "\n".join(problems)


def test_documented_public_api_exposed(package_root):
    """AGENTS.md §1.1 登记的公共 API 都必须在 __init__.py 中暴露。

    新增算子（新家族 / 新入口 / 新后端桥接）必须同时更新 AGENTS.md 的支持矩阵、
    `__init__.py` 的导出与 `_PUBLIC_API` 清单，本用例拦截漏改。
    """
    exposed = set(_exposed_names(package_root))
    missing = sorted(set(_PUBLIC_API) - exposed)
    assert not missing, (
        f"以下公共 API 未在 {_PACKAGE_NAME}/__init__.py 中暴露：{missing}"
    )


def test_version_matches_pyproject(package_root):
    """包内 __version__ 与 pyproject.toml 的 version 保持一致。"""
    pyproject = package_root.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip(f"{pyproject} 不存在，跳过版本号一致性检查")

    declared = re.search(
        r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert declared, "pyproject.toml 中未找到 version"

    init_source = (package_root / "__init__.py").read_text(encoding="utf-8")
    packed = re.search(r'^__version__\s*=\s*"([^"]+)"', init_source, re.MULTILINE)
    assert packed, "__init__.py 中未找到 __version__"
    assert declared.group(1) == packed.group(1), (
        f"版本号不同步：pyproject.toml={declared.group(1)}, "
        f"__init__.py={packed.group(1)}"
    )
