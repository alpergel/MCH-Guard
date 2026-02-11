import argparse
import csv
from pathlib import Path
from typing import Iterable, Set, Tuple


def read_pairs(csv_path: Path, rid_col: str, date_col: str) -> Set[Tuple[str, str]]:
	"""Read (RID, SCANDATE) pairs from a CSV using the provided column names.

	Skips rows where either value is missing. Trims whitespace.
	"""
	pairs: Set[Tuple[str, str]] = set()
	with csv_path.open(newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		# Normalize header keys for case-insensitive matching
		header_map = {k.lower(): k for k in reader.fieldnames or []}
		if rid_col.lower() not in header_map or date_col.lower() not in header_map:
			raise SystemExit(
				f"Columns not found in {csv_path}: expected '{rid_col}' and '{date_col}', got {reader.fieldnames}"
			)
		rid_key = header_map[rid_col.lower()]
		date_key = header_map[date_col.lower()]
		for row in reader:
			raw_rid = (row.get(rid_key) or "").strip()
			raw_date = (row.get(date_key) or "").strip()
			if not raw_rid or not raw_date:
				continue
			pairs.add((raw_rid, raw_date))
	return pairs


def write_pairs(csv_path: Path, pairs: Iterable[Tuple[str, str]]) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["RID", "SCANDATE"]) 
		for rid, date in sorted(pairs, key=lambda x: (x[0], x[1])):
			writer.writerow([rid, date])


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare (RID, SCANDATE) pairs between two CSVs.")
	parser.add_argument("file1", type=Path, help="First CSV file path")
	parser.add_argument("file2", type=Path, help="Second CSV file path")
	parser.add_argument("--file1-rid-col", default="RID", help="RID column name in file1 (default: RID)")
	parser.add_argument("--file1-date-col", default="SCANDATE", help="SCANDATE column name in file1 (default: SCANDATE)")
	parser.add_argument("--file2-rid-col", default="RID", help="RID column name in file2 (default: RID)")
	parser.add_argument("--file2-date-col", default="SCANDATE", help="SCANDATE column name in file2 (default: SCANDATE)")
	parser.add_argument(
		"--out-prefix",
		default="processed/missing_pairs",
		help="Output path prefix for CSV results (two files will be created)",
	)
	args = parser.parse_args()

	pairs1 = read_pairs(args.file1, args.file1_rid_col, args.file1_date_col)
	pairs2 = read_pairs(args.file2, args.file2_rid_col, args.file2_date_col)

	missing_in_2 = pairs1 - pairs2
	missing_in_1 = pairs2 - pairs1

	out1 = Path(f"{args.out_prefix}__{args.file1.stem}_NOT_IN_{args.file2.stem}.csv")
	out2 = Path(f"{args.out_prefix}__{args.file2.stem}_NOT_IN_{args.file1.stem}.csv")
	write_pairs(out1, missing_in_2)
	write_pairs(out2, missing_in_1)

	print(f"File1: {args.file1} -> pairs: {len(pairs1)}")
	print(f"File2: {args.file2} -> pairs: {len(pairs2)}")
	print(f"Missing in file2 (present in file1 only): {len(missing_in_2)} -> {out1}")
	print(f"Missing in file1 (present in file2 only): {len(missing_in_1)} -> {out2}")

	# Print a small sample for quick inspection
	def sample(pairs: Set[Tuple[str, str]], n: int = 10):
		return ", ".join([f"({rid}, {date})" for rid, date in list(sorted(pairs))[:n]])

	if missing_in_2:
		print("Sample missing in file2:", sample(missing_in_2))
	if missing_in_1:
		print("Sample missing in file1:", sample(missing_in_1))


if __name__ == "__main__":
	main()


