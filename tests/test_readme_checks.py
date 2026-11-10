"""Prevent the fresh-clone job from reporting success for skipped or empty suites."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('readme_report', Path(__file__).resolve().parents[1] / 'scripts/check_test_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize('outcome', ['skipped', 'failure', 'error'])
def test_incomplete_test_report_is_rejected(tmp_path, outcome):
    path = tmp_path / 'tests.xml'
    path.write_text(f'<testsuites><testsuite><testcase name="data-check"><{outcome}/></testcase></testsuite></testsuites>')
    with pytest.raises(ValueError, match='data-check'):
        report.check_report(path)


def test_empty_test_report_is_rejected(tmp_path):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuites/>')
    with pytest.raises(ValueError, match='no test cases'):
        report.check_report(path)


def test_completed_test_report_is_accepted(tmp_path):
    path = tmp_path / 'tests.xml'
    path.write_text('<testsuites><testsuite><testcase name="dataset"/><testcase name="inference"/></testsuite></testsuites>')
    assert report.check_report(path) == 2
