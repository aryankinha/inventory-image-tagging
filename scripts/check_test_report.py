"""Require a completed test run with no failures, errors, or skipped cases."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def check_report(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    if not cases:
        raise ValueError('The full-data test report contains no test cases.')
    incomplete = [case.get('name', '?') for case in cases
                  if any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped'))]
    if incomplete:
        raise ValueError('The full-data test run did not pass every case: ' + ', '.join(incomplete))
    return len(cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(f'Full-data checks: {check_report(args.report)} tests passed; no skips.')
