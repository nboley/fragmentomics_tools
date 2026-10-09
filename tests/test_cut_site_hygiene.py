"""Cross-module hygiene for the cut-site stack: ``constants`` <- ``hexamers``
<- ``simulator.measure`` <- ``simulator.draw``.

Layering, doctests across all four modules, oracle independence and removed
imports.  These span the stack, so they sit in neither module's file.  Split
out of ``tests/test_cut_site_simulator.py`` by owner decision 188; the test
bodies are unchanged.  The single-source check for the shared definitions is
``TestConstants`` in ``tests/test_hexamers.py``.

Mutations each test must catch are documented in-line as comments.
"""

import ast
import doctest
import os
import subprocess
import sys


# ── T7: Hygiene ─────────────────────────────────────────────────────────────

class TestT7Hygiene:
    """Module-level checks."""

    def test_module_doctests_execute(self):
        """M3 (lowercase in doctest).

        ``make test`` does not collect ``background_model/``, so this is the
        ONLY place these modules' doctests run.  It covers all four layers.
        ``constants`` (which pins 25/180/156) and ``hexamers`` carry every
        example today; the other two must still pass if one is added, and an
        example vanishing from either fails here rather than silently
        dropping out.
        """
        import background_model.constants as const_mod
        import background_model.simulator.measure as measure_mod
        import background_model.hexamers as hex_mod
        import background_model.simulator.draw as draw_mod
        for mod, must_have_examples in ((const_mod, True), (hex_mod, True),
                                        (measure_mod, False), (draw_mod, False)):
            results = doctest.testmod(mod, verbose=False)
            if must_have_examples:
                assert results.attempted > 0, f"no doctests found in {mod.__name__}"
            assert results.failed == 0, (
                f"{results.failed} doctest(s) failed in {mod.__name__}"
            )

    def test_oracle_is_independent(self):
        """M40 (oracle imports background_model)."""
        oracle_path = os.path.join(os.path.dirname(__file__), "cut_site_oracle.py")
        with open(oracle_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("background_model"), (
                        f"oracle imports {node.module}"
                    )
                    assert not node.module.startswith("fragmentomics_tools"), (
                        f"oracle imports {node.module}"
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("background_model")
                        assert not alias.name.startswith("fragmentomics_tools")

    def test_no_removed_feature_imports(self):
        """M40 (module imports simulator.precompute).

        Also: ``background_model.cut_site_stats`` is gone, not shimmed. Owner
        decision 187 moved it to ``simulator.measure`` with no alias left
        behind, so an old import fails loudly instead of resolving.
        """
        import importlib.util
        assert importlib.util.find_spec("background_model.cut_site_stats") is None
        import background_model.simulator.measure as measure_mod
        import background_model.hexamers as hex_mod
        import background_model.simulator.draw as draw_mod
        banned = {"flgc", "simulator.capture", "simulator.precompute",
                   "simulator.weights", "simulator.sampler", "simulator.emit"}
        for mod in (hex_mod, measure_mod, draw_mod):
            with open(mod.__file__) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for b in banned:
                        assert b not in node.module, (
                            f"{mod.__name__} imports removed feature: "
                            f"{node.module}"
                        )

    def test_layering_holds(self):
        """Owner decision 169 split one module into three layers, each
        importing only from layers above it: ``hexamers.py`` (numpy + stdlib
        only) <- ``simulator/measure.py`` (adds pandas, fragmentomics_tools) <-
        ``simulator/draw.py``. Decisions 174-177 put ``constants.py`` (stdlib
        + ``background_model.tracks`` only) above all three, and decision 187
        moved ``measure`` (formerly ``cut_site_stats.py``) into the simulator
        package. Guards mutations L1 (hexamers imports pandas), L2 (measure
        imports upward from draw), L3 (constants imports numpy) and L4
        (measure imports draw relatively, ``from . import draw``).

        "Above", not "the one above": ``draw`` importing ``hexamers`` directly
        is allowed, so this checks each import against the importer's own
        layer rather than demanding it come from the adjacent one.
        """
        import background_model.constants as const_mod
        import background_model.hexamers as hex_mod
        import background_model.simulator as sim_pkg
        import background_model.simulator.draw as draw_mod
        import background_model.simulator.measure as measure_mod

        # Upstream first: a module may import only from a LOWER index here,
        # i.e. from a layer above it. measure is the one simulator module above
        # the draw; any OTHER background_model.simulator submodule is the
        # draw's layer, so a new sibling cannot slip in above measure unseen.
        measure_layer = "background_model.simulator.measure"
        sim_pkg_name = "background_model.simulator"
        layers = ["background_model.constants", "background_model.hexamers",
                  measure_layer, sim_pkg_name]

        def layer_of(name):
            # The bare package is a namespace, not a layer: measure and draw
            # both live in it, so `from . import x` names it as the base. The
            # alias (``.x``) still carries the layer. Skipping the bare name is
            # sound only while the package __init__ imports nothing, which is
            # asserted below.
            if name == sim_pkg_name:
                return None
            for i, prefix in enumerate(layers):
                if name == prefix or name.startswith(prefix + "."):
                    return i
            return None

        def imported_names(mod):
            # Relative imports resolve against the module's own package.
            package = mod.__name__.rsplit(".", 1)[0]
            with open(mod.__file__) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        yield alias.name
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        parts = package.split(".")
                        parts = parts[:len(parts) - (node.level - 1)]
                        base = ".".join(parts + ([base] if base else []))
                    yield base
                    # `from background_model import simulator` names the
                    # layer in the alias, not in the module.
                    for alias in node.names:
                        yield f"{base}.{alias.name}"

        with open(sim_pkg.__file__) as f:
            assert not [n for n in ast.walk(ast.parse(f.read()))
                        if isinstance(n, (ast.Import, ast.ImportFrom))], (
                f"{sim_pkg.__file__} imports something; the bare-package skip "
                f"in layer_of is no longer sound")

        for own, mod in enumerate((const_mod, hex_mod, measure_mod, draw_mod)):
            cross = set()
            for name in imported_names(mod):
                other = layer_of(name)
                if other is None or other == own:
                    continue
                assert other < own, (
                    f"{mod.__name__} imports {name} from a layer below it"
                )
                cross.add(other)
            if own:
                # Non-vacuity: the walker must see the real imports these
                # modules make, or the assertion above checks nothing.
                assert cross, f"{mod.__name__}: no cross-layer import seen"

        # Third-party/first-party imports each light module may make: top-level
        # packages, plus exact background_model modules. Anything else is
        # outside its layer.
        stdlib = set(sys.stdlib_module_names) | {"__future__"}
        light = (
            (const_mod, stdlib, {"background_model.tracks"},
             "stdlib+tracks"),
            (hex_mod, stdlib | {"numpy"}, {"background_model.constants"},
             "numpy+stdlib+constants"),
        )
        for mod, allowed_tops, allowed_exact, what in light:
            fname = os.path.basename(mod.__file__)
            with open(mod.__file__) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    assert (name.split(".")[0] in allowed_tops
                            or name in allowed_exact), (
                        f"{fname} imports {name}, outside the {what} layer"
                    )

        # Importing only hexamers must not pull in pandas, torch or
        # fragmentomics_tools, and importing only constants must not pull in
        # numpy either. background_model/__init__.py imports
        # background_model.config, which imports only stdlib and
        # background_model.tracks (stdlib only). That is why this holds; the
        # subprocess checks it rather than assuming it.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for module, extra_banned in (("background_model.hexamers", ()),
                                     ("background_model.constants", ("numpy",))):
            r = subprocess.run(
                [sys.executable, "-c",
                 f"import {module}, sys\n"
                 f"extra = {tuple(extra_banned)!r}\n"
                 "banned = [m for m in sys.modules if m == 'pandas' or m == 'torch' "
                 "or m.startswith('fragmentomics_tools') or m in extra]\n"
                 "assert not banned, banned\n"],
                capture_output=True, text=True, cwd=repo_root,
            )
            assert r.returncode == 0, (
                f"importing {module} alone pulled in a "
                f"forbidden module:\n{r.stdout}\n{r.stderr}"
            )
