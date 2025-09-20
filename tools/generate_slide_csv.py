"""Utility to build slide-level metadata CSVs for the PAMIL pipelines.

The generated CSV follows the format consumed by :mod:`dataset_generic_npy`
(and the HDF5 equivalent): each row describes a single slide with the
following columns:

``case_id``
    Identifier of the patient/case. By default this mirrors the slide ID,
    but it can be derived from the parent directory or a custom regular
    expression when needed.
``slide_id``
    Stem of the feature file, optionally post-processed by a regex.
``label``
    Categorical label stored as a string. At training time this value is
    mapped to an integer via the ``label_dict`` argument of the dataset
    loader.

The script is intentionally flexible so it can accommodate feature exports
that follow different naming conventions. Common usage patterns:

* Prefix-based labels (``FA 47 B1.h5`` → label ``FA``)::

      python tools/generate_slide_csv.py \
          --features-dir /path/to/feats_h5 \
          --output-csv dataset_csv/custom.csv

* Custom mapping (``FA`` → ``0``)::

      python tools/generate_slide_csv.py \
          --features-dir /path/to/feats_h5 \
          --output-csv dataset_csv/custom.csv \
          --label-map FA=0 PT=1

* Regex-derived identifiers::

      python tools/generate_slide_csv.py \
          --features-dir /path/to/feats_h5 \
          --output-csv dataset_csv/custom.csv \
          --label-source regex --label-regex "^(?P<label>[A-Z]+)" \
          --case-source regex --case-regex "^(?P<case_id>[A-Z]+\\s?\\d+)"
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _normalise_extension(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        raise ValueError("File extension cannot be empty.")
    if not ext.startswith('.'):
        ext = f".{ext}"
    return ext.lower()


def _parse_mapping(pairs: Optional[Iterable[str]]) -> Dict[str, str]:
    if not pairs:
        return {}
    mapping: Dict[str, str] = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"Invalid mapping entry '{pair}'. Expected KEY=VALUE format.")
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid mapping entry '{pair}'.")
        mapping[key] = value
    return mapping


def _split_token(text: str) -> str:
    tokens = re.split(r"[\s_-]+", text)
    return tokens[0] if tokens else text


def _regex_extract(pattern: str, text: str, group: str) -> str:
    match = re.search(pattern, text)
    if not match or group not in match.groupdict():
        raise ValueError(
            f"Pattern '{pattern}' with group '{group}' did not match text '{text}'."
        )
    return str(match.group(group))


def _derive_value(source: str, *, stem: str, path: Path, regex: Optional[str], group: str) -> str:
    if source == 'stem':
        return stem
    if source == 'prefix':
        return _split_token(stem)
    if source == 'parent':
        return path.parent.name
    if source == 'regex':
        if not regex:
            raise ValueError(f"A --{group.replace('_', '-')} regex must be provided when using 'regex' source.")
        return _regex_extract(regex, stem, group)
    raise ValueError(f"Unsupported source '{source}'.")


def build_metadata(
    features_dir: Path,
    *,
    extension: str,
    recursive: bool,
    label_source: str,
    case_source: str,
    label_regex: Optional[str],
    case_regex: Optional[str],
    label_map: Dict[str, str],
    include_path: bool,
) -> List[Dict[str, str]]:
    if not features_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {features_dir}")

    pattern = f"*{extension}"
    files = (
        sorted(features_dir.rglob(pattern))
        if recursive
        else sorted(p for p in features_dir.glob(pattern) if p.is_file())
    )

    if not files:
        raise FileNotFoundError(
            f"No feature files with extension '{extension}' found under {features_dir}."
        )

    records: List[Dict[str, str]] = []
    for path in files:
        stem = path.stem
        slide_id = stem

        label_token = _derive_value(
            label_source,
            stem=stem,
            path=path,
            regex=label_regex,
            group='label',
        )
        label = label_map.get(label_token, label_token)

        case_id = _derive_value(
            case_source,
            stem=stem,
            path=path,
            regex=case_regex,
            group='case_id',
        )

        record = {
            'case_id': case_id,
            'slide_id': slide_id,
            'label': label,
        }

        if include_path:
            record['path'] = str(path.relative_to(features_dir))

        records.append(record)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, required=True,
                        help='Directory containing one feature file per slide.')
    parser.add_argument('--output-csv', type=Path, required=True,
                        help='Destination CSV file (directories are created automatically).')
    parser.add_argument('--extension', type=str, default='.h5',
                        help='Feature file extension (default: .h5).')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for feature files in subdirectories.')
    parser.add_argument('--label-source', choices=['prefix', 'stem', 'parent', 'regex'], default='prefix',
                        help='How to derive the slide label from the filename (default: prefix).')
    parser.add_argument('--case-source', choices=['stem', 'parent', 'regex'], default='stem',
                        help='How to derive the case identifier (default: stem).')
    parser.add_argument('--label-regex', type=str, default=None,
                        help="Regular expression with a named 'label' group used when --label-source=regex.")
    parser.add_argument('--case-regex', type=str, default=None,
                        help="Regular expression with a named 'case_id' group used when --case-source=regex.")
    parser.add_argument('--label-map', nargs='*', default=None,
                        help='Optional KEY=VALUE pairs used to remap derived labels (e.g. FA=0 PT=1).')
    parser.add_argument('--include-path', action='store_true',
                        help='Include a column with the relative path to each feature file.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extension = _normalise_extension(args.extension)
    label_map = _parse_mapping(args.label_map)

    records = build_metadata(
        args.features_dir,
        extension=extension,
        recursive=args.recursive,
        label_source=args.label_source,
        case_source=args.case_source,
        label_regex=args.label_regex,
        case_regex=args.case_regex,
        label_map=label_map,
        include_path=args.include_path,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['case_id', 'slide_id', 'label']
    if args.include_path:
        fieldnames.append('path')

    with args.output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} entries to {args.output_csv}.")


if __name__ == '__main__':
    main()
