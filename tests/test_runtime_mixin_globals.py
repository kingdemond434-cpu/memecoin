"""Desk mixins resolve their globals in their own module, not main.py's.

Spot logged ``name 'AgeBandedPredictor' is not defined`` from every
intelligence sweep (training.py _reload_promoted_model) for days before it
died on 2026-11-03: main.py imported the name, the mixin module did not.
"""
import ast
import builtins
import importlib
from pathlib import Path

import pytest

MIXINS = ["src.runtime.training", "src.runtime.maintenance"]


def _free_global_loads(tree):
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None:
                        bound.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loads - bound - set(dir(builtins))


@pytest.mark.parametrize("name", MIXINS)
def test_mixin_has_no_unresolved_globals(name):
    module = importlib.import_module(name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    missing = sorted(n for n in _free_global_loads(tree) if n not in vars(module))
    assert missing == []


def test_reload_promoted_model_resolves_predictor():
    from src.runtime import training
    from src.strategies.age_banded import AgeBandedPredictor
    assert training.AgeBandedPredictor is AgeBandedPredictor
