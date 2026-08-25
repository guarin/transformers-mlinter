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


import unittest

from tests.rule_test_utils import Path, _helpers_mod, _trf060_mod, date, mlinter, patch, tempfile


class TRF060Test(unittest.TestCase):
    # --- TRF060: package-internal imports must be relative ---

    def test_trf060_flags_absolute_package_imports(self):
        source = """
from transformers.utils import logging
import os, transformers.models.auto as auto_module
from transformers import AutoModel
import transformers
"""
        file_path = Path("src/transformers/utility.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 4)
        self.assertIn("`transformers.utils` is imported", violations[0].message)
        self.assertIn("`transformers.models.auto` is imported", violations[1].message)
        self.assertTrue(all(violation.line_number in {2, 3, 4, 5} for violation in violations))

    def test_trf060_accepts_relative_and_external_imports(self):
        source = """
from .utils import logging
from ..models.auto import AutoModel
import torch
"""
        file_path = Path("src/transformers/utility.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(violations, [])

    def test_trf060_respects_suppression(self):
        source = """
# trf-ignore: TRF060
from transformers.utils import logging
"""
        file_path = Path("src/transformers/utility.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(violations, [])

    def test_trf060_skips_model_conversion_scripts_without_requiring_a_main_guard(self):
        source = "from transformers.utils import logging\nconvert()\n"
        file_path = Path("src/transformers/models/foo/convert_foo.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(violations, [])

    def test_trf060_skips_guarded_root_conversion_scripts(self):
        source = """from transformers.utils import logging

if "__main__" == __name__:
    main()
"""
        file_path = Path("src/transformers/convert_checkpoints.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(violations, [])

    def test_trf060_keeps_importable_root_convert_modules_in_scope(self):
        source = "from transformers.utils import logging\n"
        file_path = Path("src/transformers/convert_slow_tokenizer.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 1)

    def test_trf060_does_not_exempt_non_conversion_files_with_a_main_guard(self):
        source = """from transformers.utils import logging

if __name__ == "__main__":
    main()
"""
        file_path = Path("src/transformers/utility.py")
        violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 1)

    def test_trf060_skips_cli_files(self):
        source = "from transformers.utils import logging\n"
        for relative_path in ("chat.py", "transformers.py", "serve.py", "serving/utils.py"):
            with self.subTest(relative_path=relative_path):
                file_path = Path("src/transformers/cli") / relative_path
                violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
                self.assertEqual(violations, [])

    def test_trf060_cutoff_exempts_old_non_model_files_by_copyright_year(self):
        source = """# Copyright 2025-present The HuggingFace Team. All rights reserved.
from transformers.utils import logging
"""
        file_path = Path("src/transformers/utility.py")
        with patch.object(_trf060_mod, "CUTOFF_DATE", "2026-06-20"):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(violations, [])

    def test_trf060_cutoff_checks_files_whose_copyright_year_matches_the_cutoff_year(self):
        # A copyright year equal to the cutoff year cannot say which side of the cutoff the file
        # landed on, so the file stays checked rather than being grandfathered on a guess.
        source = """# Copyright (C) 2026 The HuggingFace Team. All rights reserved.
from transformers.utils import logging
"""
        file_path = Path("src/transformers/utility.py")
        with patch.object(_trf060_mod, "CUTOFF_DATE", "2026-06-20"):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 1)

    def test_trf060_cutoff_checks_files_with_a_newer_copyright_year(self):
        source = """# Copyright 2027 The HuggingFace Team. All rights reserved.
from transformers.utils import logging
"""
        file_path = Path("src/transformers/utility.py")
        with patch.object(_trf060_mod, "CUTOFF_DATE", "2026-06-20"):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 1)

    def test_trf060_cutoff_checks_files_without_a_copyright_year(self):
        source = """# Copyright The HuggingFace Team. All rights reserved.
from transformers.utils import logging
"""
        file_path = Path("src/transformers/utility.py")
        with patch.object(_trf060_mod, "CUTOFF_DATE", "2026-06-20"):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 1)

    def test_trf060_cutoff_uses_model_contribution_date_inside_models(self):
        # The old copyright year must not win over the model's documented contribution date.
        source = """# Copyright 2020 The HuggingFace Team. All rights reserved.
from transformers.utils import logging
"""
        file_path = Path("src/transformers/models/foo/modeling_foo.py")
        with (
            patch.object(_trf060_mod, "CUTOFF_DATE", "2026-06-20"),
            patch.object(_helpers_mod, "model_contribution_date", return_value=date(2026, 6, 20)),
        ):
            violations = mlinter.analyze_file(file_path, source, enabled_rules={mlinter.TRF060})
        self.assertEqual(len(violations), 1)

    def test_trf060_discovery_scans_src_but_skips_tests_and_generated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_root = root / "src/transformers"
            tests_root = root / "tests"
            src_root.mkdir(parents=True)
            tests_root.mkdir()
            source = src_root / "utility.py"
            generated = src_root / "generated_utility.py"
            test = tests_root / "test_utility.py"
            source.write_text("", encoding="utf-8")
            generated.write_text(f"# {_helpers_mod.GENERATED_FILE_MARKER} utility.py\n", encoding="utf-8")
            test.write_text("", encoding="utf-8")

            with patch.object(mlinter, "SRC_ROOT", src_root):
                found = dict(mlinter.iter_files({mlinter.TRF060}))

        self.assertEqual(found, {source: {mlinter.TRF060}})
