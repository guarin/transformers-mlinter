# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRF060: package-internal imports must be relative."""

import ast
from pathlib import Path

from ._helpers import Violation, _has_rule_suppression, is_exempt_by_cutoff


RULE_ID = ""  # Set by discovery
CUTOFF_DATE = ""  # Set by discovery from rules.toml cutoff_date; empty means no exemption
PACKAGE_NAME = "transformers"


def _is_package_import(module_name: str) -> bool:
    return module_name == PACKAGE_NAME or module_name.startswith(f"{PACKAGE_NAME}.")


def _is_name_main_comparison(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
        return False
    if len(node.comparators) != 1:
        return False
    left, right = node.left, node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _has_main_guard(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.If) and _is_name_main_comparison(node.test) for node in ast.walk(tree))


def _is_conversion_script(tree: ast.Module, file_path: Path) -> bool:
    if not file_path.name.startswith("convert_"):
        return False
    # Model conversion files are standalone scripts, including a few legacy files that execute at
    # module scope without a main guard. At the package root, keep importable helpers such as
    # convert_slow_tokenizer.py in scope and exempt only an explicitly executable script.
    return "models" in file_path.parts or _has_main_guard(tree)


def _is_cli_file(file_path: Path) -> bool:
    # TODO: Move this exemption to rules.toml if mlinter gains configurable path exclusions.
    return "cli" in file_path.parts


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if (
        _is_conversion_script(tree, file_path)
        or _is_cli_file(file_path)
        or is_exempt_by_cutoff(file_path, CUTOFF_DATE, source_lines)
    ):
        return []

    violations: list[Violation] = []
    for node in ast.walk(tree):
        imported_modules: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None and _is_package_import(node.module):
                imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names if _is_package_import(alias.name))

        if not imported_modules or _has_rule_suppression(source_lines, RULE_ID, node.lineno):
            continue

        for module_name in imported_modules:
            violations.append(
                Violation(
                    file_path=file_path,
                    line_number=node.lineno,
                    message=(
                        f"{RULE_ID}: `{module_name}` is imported through the package's absolute name. "
                        "Use a relative import for references within `transformers`."
                    ),
                )
            )

    return violations
