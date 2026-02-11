import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


def load_dataset(csv_path: Path) -> pd.DataFrame:
	"""Load the MCH dataset CSV."""
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV not found at: {csv_path}")
	# Read as default dtypes; parse SCANDATE as date for reliable grouping
	df = pd.read_csv(csv_path, parse_dates=["SCANDATE"], infer_datetime_format=True)
	return df


def compute_per_date_counts(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute per-date MCH_count and MCH_pos_flag per (RID, SCANDATE date).

	- MCH_count: number of rows where TYPE == 'MCH' for that RID on that date,
	  or taken from an existing 'MCH_count' column if TYPE is not present.
	- MCH_pos_flag: 0/1 indicator. Prefer existing 'MCH_pos' if present
	  (aggregated with max). If absent, fall back to 'NOFINDINGS' by
	  setting MCH_pos_flag = 0 when NOFINDINGS == 1 else 1. If neither is
	  available, derive MCH_pos_flag = 0 if MCH_count == 0 else 1.
	"""
	if not {"RID", "SCANDATE"}.issubset(df.columns):
		raise KeyError("Missing required columns: ['RID', 'SCANDATE']")

	work = df.copy()
	work["SCANDATE"] = pd.to_datetime(work["SCANDATE"], errors="coerce").dt.date

	# Compute MCH_count
	if "MCH_count" in work.columns:
		mch_counts = (
			work.groupby(["RID", "SCANDATE"], as_index=False)["MCH_count"]
			.max()
		)
	elif "TYPE" in work.columns:
		mch_counts = (
			work.assign(is_mch=(work["TYPE"] == "MCH"))
			.groupby(["RID", "SCANDATE"], as_index=False)["is_mch"]
			.sum()
			.rename(columns={"is_mch": "MCH_count"})
		)
	else:
		raise KeyError("Missing columns to compute MCH_count: need either 'MCH_count' or 'TYPE'")

	# Compute MCH_pos_flag preference order: MCH_pos -> NOFINDINGS -> from MCH_count
	if "MCH_pos" in work.columns:
		pos_series = (
			work.assign(MCH_pos_num=pd.to_numeric(work["MCH_pos"], errors="coerce").fillna(0))
			.groupby(["RID", "SCANDATE"], as_index=False)["MCH_pos_num"]
			.max()
			.rename(columns={"MCH_pos_num": "MCH_pos_flag"})
		)
	elif "NOFINDINGS" in work.columns:
		pos_series = (
			work.assign(NOFINDINGS_NUM=pd.to_numeric(work["NOFINDINGS"], errors="coerce").fillna(0))
			.groupby(["RID", "SCANDATE"], as_index=False)["NOFINDINGS_NUM"]
			.max()
			.assign(MCH_pos_flag=lambda d: (d["NOFINDINGS_NUM"] == 0).astype(int))
			.drop(columns=["NOFINDINGS_NUM"])
		)
	else:
		pos_series = (
			mch_counts.assign(MCH_pos_flag=(mch_counts["MCH_count"] > 0).astype(int))
			[["RID", "SCANDATE", "MCH_pos_flag"]]
		)

	merged = mch_counts.merge(pos_series, on=["RID", "SCANDATE"], how="left").fillna({"MCH_count": 0, "MCH_pos_flag": 0})
	merged["MCH_count"] = pd.to_numeric(merged["MCH_count"], errors="coerce").fillna(0).astype(int)
	merged["MCH_pos_flag"] = pd.to_numeric(merged["MCH_pos_flag"], errors="coerce").fillna(0).astype(int)

	# Enforce rule: MCH_pos == 0 implies MCH_count == 0
	merged.loc[merged["MCH_pos_flag"] == 0, "MCH_count"] = 0
	return merged


def bucket_label(mch_pos_flag: int, _mch_count: int) -> str:
	"""Map a per-date record to a binary MCH status using MCH_pos_flag.

	Labels:
	- No MCH
	- MCH
	"""
	if int(mch_pos_flag) == 0:
		return "No MCH"
	return "MCH"


def baseline_last_buckets(per_date: pd.DataFrame) -> pd.DataFrame:
	"""For each RID, determine baseline and last bucket labels.

	If there is only one datapoint for a RID, the last bucket is 'NODATA'.
	"""
	per_date_sorted = per_date.sort_values(["RID", "SCANDATE"]).copy()

	# Compute baseline (first) and last (last) per RID
	first_rows = per_date_sorted.groupby("RID", as_index=False).first()
	last_rows = per_date_sorted.groupby("RID", as_index=False).last()

	first_rows["baseline_bucket"] = [
		bucket_label(pos, cnt) for pos, cnt in zip(first_rows["MCH_pos_flag"], first_rows["MCH_count"])
	]
	last_rows["last_bucket_raw"] = [
		bucket_label(pos, cnt) for pos, cnt in zip(last_rows["MCH_pos_flag"], last_rows["MCH_count"])
	]

	merged = first_rows[["RID", "SCANDATE", "baseline_bucket"]].merge(
		last_rows[["RID", "SCANDATE", "last_bucket_raw"]], on="RID", suffixes=("_base", "_last"), how="left"
	)

	# Determine if only one datapoint
	merged["single_visit"] = merged["SCANDATE_base"] == merged["SCANDATE_last"]
	merged["last_bucket"] = merged["last_bucket_raw"].where(~merged["single_visit"], other="NODATA")
	return merged[["RID", "baseline_bucket", "last_bucket", "single_visit"]]


def flow_counts(buckets_df: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate counts for flows baseline_bucket -> last_bucket."""
	flows = (
		buckets_df.groupby(["baseline_bucket", "last_bucket"], as_index=False)
		.size()
		.rename(columns={"size": "count"})
	)
	return flows


def draw_two_column_river(flows: pd.DataFrame, left_order: List[str], right_order: List[str], colors: Dict[str, str], output_path: Path) -> Path:
	"""Draw a simple two-column alluvial (river) plot using matplotlib.

	Flows are colored by the left (baseline) bucket.
	"""
	# Publication-style defaults
	plt.rcParams.update({
		"font.size": 11,
		"font.family": "DejaVu Sans",
		"axes.titlesize": 13,
		"figure.dpi": 300,
	})
	# Layout parameters
	left_x = 0.12
	right_x = 0.88
	curve_x_offset = 0.28
	bar_width = 0.05
	fig_w, fig_h, dpi = 8.5, 5.5, 300

	# Compute totals for stacking
	left_totals = flows.groupby("baseline_bucket")["count"].sum().reindex(left_order).fillna(0).astype(int)
	right_totals = flows.groupby("last_bucket")["count"].sum().reindex(right_order).fillna(0).astype(int)
	max_total = max(left_totals.sum(), right_totals.sum())

	# Compute y extents (top to bottom) normalized to total height
	def stack_positions(totals: pd.Series) -> Dict[str, Tuple[float, float]]:
		positions: Dict[str, Tuple[float, float]] = {}
		current_y = 0.98
		total_sum = totals.sum() if totals.sum() > 0 else 1
		for label, val in totals.items():
			height = (val / total_sum) * 0.9
			positions[label] = (current_y - height, current_y)
			current_y -= height + 0.02
		return positions

	left_pos = stack_positions(left_totals)
	right_pos = stack_positions(right_totals)

	# Initialize running offsets within each category to place sub-flows
	left_offsets = {k: left_pos[k][0] for k in left_order}
	right_offsets = {k: right_pos[k][0] for k in right_order}

	fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
	ax.axis("off")

	# Draw category bars
	for label in left_order:
		(y0, y1) = left_pos[label]
		ax.add_patch(Rectangle((left_x - bar_width / 2, y0), bar_width, y1 - y0, color="#E0E0E0"))
		ax.text(left_x - 0.07, (y0 + y1) / 2, f"{label}\n(n={left_totals.get(label, 0)})", va="center", ha="right", fontsize=11)

	for label in right_order:
		(y0, y1) = right_pos[label]
		ax.add_patch(Rectangle((right_x - bar_width / 2, y0), bar_width, y1 - y0, color="#E0E0E0"))
		ax.text(right_x + 0.07, (y0 + y1) / 2, f"{label}\n(n={right_totals.get(label, 0)})", va="center", ha="left", fontsize=11)

	# Draw flows
	for _, row in flows.iterrows():
		l_label = row["baseline_bucket"]
		r_label = row["last_bucket"]
		count = int(row["count"])
		if count <= 0:
			continue

		# Heights relative to left and right stacks
		left_total_sum = left_totals.sum() if left_totals.sum() > 0 else 1
		right_total_sum = right_totals.sum() if right_totals.sum() > 0 else 1
		flow_height_left = (count / left_total_sum) * 0.9
		flow_height_right = (count / right_total_sum) * 0.9

		# Use min of the two heights for a neat band
		height = min(flow_height_left, flow_height_right)

		# Determine vertical span for this sub-flow on left and right
		y0_left = left_offsets[l_label]
		y1_left = y0_left + height
		left_offsets[l_label] = y1_left

		y0_right = right_offsets[r_label]
		y1_right = y0_right + height
		right_offsets[r_label] = y1_right

		# Path for the band (top and bottom Bezier curves)
		verts = [
			(left_x, y0_left),
			(left_x + curve_x_offset, y0_left),
			(right_x - curve_x_offset, y0_right),
			(right_x, y0_right),
			(right_x, y1_right),
			(right_x - curve_x_offset, y1_right),
			(left_x + curve_x_offset, y1_left),
			(left_x, y1_left),
			(left_x, y0_left),
		]
		codes = [
			MplPath.MOVETO,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.LINETO,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CURVE4,
			MplPath.CLOSEPOLY,
		]
		path = MplPath(verts, codes)
		color = colors.get(l_label, "#5B84B1")
		patch = PathPatch(path, facecolor=color, edgecolor="none", alpha=0.7)
		ax.add_patch(patch)

	# Title
	ax.text(0.5, 1.02, "Baseline to last MCH status ", ha="center", va="bottom", fontsize=13)
	ax.text(left_x, 1.0, "Baseline", ha="center", va="bottom", fontsize=12)
	ax.text(right_x, 1.0, "Last", ha="center", va="bottom", fontsize=12)

	# Legend (color keyed by baseline status)
	handles = [
		Rectangle((0, 0), 1, 1, facecolor=colors.get(label, "#5B84B1"), edgecolor="none", alpha=0.7, label=label)
		for label in left_order
	]
	ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=len(left_order), frameon=False)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	# Also export high-quality PDF for publication
	pdf_path = output_path.with_suffix(".pdf")
	fig.savefig(pdf_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def analyze_mch_negative_progression(per_date: pd.DataFrame, buckets: pd.DataFrame) -> Dict:
	"""Analyze MCH-negative patients at baseline for follow-up and progression statistics.
	
	Returns:
		Dictionary with:
		- pct_with_followup: percentage with follow-up data
		- pct_developed_mch: percentage who developed MCH among those with follow-up
		- followup_years_mean: mean follow-up duration in years
		- followup_years_std: standard deviation of follow-up duration
	"""
	import numpy as np
	
	# Filter for MCH-negative at baseline
	mch_neg_baseline = buckets[buckets["baseline_bucket"] == "No MCH"].copy()
	total_mch_neg = len(mch_neg_baseline)
	
	# Calculate percentage with follow-up data
	with_followup = mch_neg_baseline[~mch_neg_baseline["single_visit"]]
	n_with_followup = len(with_followup)
	pct_with_followup = (n_with_followup / total_mch_neg * 100) if total_mch_neg > 0 else 0
	
	# Among those with follow-up, calculate who developed MCH
	# Developed MCH = last_bucket is NOT "No MCH" and NOT "NODATA"
	developed_mch = with_followup[
		(with_followup["last_bucket"] != "No MCH") & 
		(with_followup["last_bucket"] != "NODATA")
	]
	n_developed_mch = len(developed_mch)
	pct_developed_mch = (n_developed_mch / n_with_followup * 100) if n_with_followup > 0 else 0
	
	# Calculate follow-up duration in years for those with follow-up
	# Merge with per_date to get actual dates
	per_date_sorted = per_date.sort_values(["RID", "SCANDATE"]).copy()
	
	# Get baseline and last scan dates for each RID
	baseline_dates = per_date_sorted.groupby("RID", as_index=False).first()[["RID", "SCANDATE"]].rename(columns={"SCANDATE": "baseline_date"})
	last_dates = per_date_sorted.groupby("RID", as_index=False).last()[["RID", "SCANDATE"]].rename(columns={"SCANDATE": "last_date"})
	
	dates_merged = baseline_dates.merge(last_dates, on="RID")
	dates_merged["baseline_date"] = pd.to_datetime(dates_merged["baseline_date"])
	dates_merged["last_date"] = pd.to_datetime(dates_merged["last_date"])
	dates_merged["followup_days"] = (dates_merged["last_date"] - dates_merged["baseline_date"]).dt.days
	dates_merged["followup_years"] = dates_merged["followup_days"] / 365.25
	
	# Merge with MCH-neg baseline patients who have follow-up
	with_followup_dates = with_followup.merge(dates_merged, on="RID", how="left")
	
	followup_years = with_followup_dates["followup_years"].dropna()
	followup_years_mean = followup_years.mean() if len(followup_years) > 0 else 0
	followup_years_std = followup_years.std() if len(followup_years) > 0 else 0
	
	return {
		"total_mch_neg_baseline": total_mch_neg,
		"n_with_followup": n_with_followup,
		"pct_with_followup": pct_with_followup,
		"n_developed_mch": n_developed_mch,
		"pct_developed_mch": pct_developed_mch,
		"followup_years_mean": followup_years_mean,
		"followup_years_std": followup_years_std,
	}


def subgroup_longitudinal_summary(per_date: pd.DataFrame, buckets: pd.DataFrame, baseline_label: str) -> Dict:
	"""Compute longitudinal metrics for a baseline subgroup ("No MCH" or "MCH").

	Metrics returned (keys):
	- n_total: total subjects in subgroup
	- n_with_followup, pct_with_followup
	- mean_followup_assess, sd_followup_assess (number of follow-up assessments = visits - 1)
	- mean_followup_days, sd_followup_days
	- n_mch_after_baseline, pct_mch_after_baseline (any MCH at follow-up)
	"""
	per_date_sorted = per_date.sort_values(["RID", "SCANDATE"]).copy()
	subject_rids = buckets.loc[buckets["baseline_bucket"] == baseline_label, "RID"].unique()
	if len(subject_rids) == 0:
		return {
			"baseline_label": baseline_label,
			"n_total": 0,
			"n_with_followup": 0,
			"pct_with_followup": 0.0,
			"mean_followup_assess": 0.0,
			"sd_followup_assess": 0.0,
			"mean_followup_days": 0.0,
			"sd_followup_days": 0.0,
			"n_mch_after_baseline": 0,
			"pct_mch_after_baseline": 0.0,
		}

	sub = per_date_sorted[per_date_sorted["RID"].isin(subject_rids)].copy()
	counts = sub.groupby("RID").size().rename("num_visits").reset_index()
	dates = sub.groupby("RID", as_index=False).agg(
		baseline_date=("SCANDATE", "first"),
		last_date=("SCANDATE", "last"),
	)
	dates["baseline_date"] = pd.to_datetime(dates["baseline_date"])
	dates["last_date"] = pd.to_datetime(dates["last_date"])
	dates["followup_days"] = (dates["last_date"] - dates["baseline_date"]).dt.days

	merged = counts.merge(dates, on="RID", how="left")
	merged["num_followup_assessments"] = (merged["num_visits"] - 1).clip(lower=0)
	merged["has_followup"] = merged["num_visits"] > 1

	# Any MCH present at follow-up (after the baseline row)
	def mch_after_baseline(group: pd.DataFrame) -> int:
		if len(group) <= 1:
			return 0
		return int(pd.to_numeric(group.iloc[1:]["MCH_pos_flag"], errors="coerce").fillna(0).max() > 0)

	# Compute indicator as a proper DataFrame, robust across pandas versions
	after = sub.groupby("RID").apply(mch_after_baseline).reset_index(name="mch_after_baseline")
	after["mch_after_baseline"] = pd.to_numeric(after["mch_after_baseline"], errors="coerce").fillna(0).astype(int)
	merged = merged.merge(after, on="RID", how="left")
	merged["mch_after_baseline"] = pd.to_numeric(merged["mch_after_baseline"], errors="coerce").fillna(0).astype(int)

	n_total = len(subject_rids)
	with_followup = merged[merged["has_followup"]]
	n_with_followup = len(with_followup)
	pct_with_followup = (n_with_followup / n_total * 100) if n_total > 0 else 0.0

	mean_followup_assess = with_followup["num_followup_assessments"].mean() if n_with_followup > 0 else 0.0
	sd_followup_assess = with_followup["num_followup_assessments"].std() if n_with_followup > 0 else 0.0
	mean_followup_days = with_followup["followup_days"].mean() if n_with_followup > 0 else 0.0
	sd_followup_days = with_followup["followup_days"].std() if n_with_followup > 0 else 0.0

	n_mch_after = int(with_followup["mch_after_baseline"].sum())
	pct_mch_after = (n_mch_after / n_with_followup * 100) if n_with_followup > 0 else 0.0

	return {
		"baseline_label": baseline_label,
		"n_total": n_total,
		"n_with_followup": n_with_followup,
		"pct_with_followup": pct_with_followup,
		"mean_followup_assess": float(mean_followup_assess) if pd.notna(mean_followup_assess) else 0.0,
		"sd_followup_assess": float(sd_followup_assess) if pd.notna(sd_followup_assess) else 0.0,
		"mean_followup_days": float(mean_followup_days) if pd.notna(mean_followup_days) else 0.0,
		"sd_followup_days": float(sd_followup_days) if pd.notna(sd_followup_days) else 0.0,
		"n_mch_after_baseline": n_mch_after,
		"pct_mch_after_baseline": pct_mch_after,
	}


def main(csv_path: str | None = None) -> None:
	base_dir = Path(__file__).resolve().parents[2]
	default_csv = base_dir / "processed" / "merge.csv"
	csv_file = Path(csv_path) if csv_path else default_csv

	df = load_dataset(csv_file)
	per_date = compute_per_date_counts(df)
	buckets = baseline_last_buckets(per_date)
	flows = flow_counts(buckets)

	# Define binary category orders and colors
	left_order = [
		"No MCH",
		"MCH",
	]
	right_order = [
		"No MCH",
		"MCH",
		"NODATA",
	]
	colors = {
		"No MCH": "#6BA292",
		"MCH": "#5B84B1",
	}

	# Ensure flows include only categories present in orders
	flows["baseline_bucket"] = flows["baseline_bucket"].where(flows["baseline_bucket"].isin(left_order))
	flows["last_bucket"] = flows["last_bucket"].where(flows["last_bucket"].isin(right_order))

	# Sort flows for stable stacking
	flows = flows.sort_values(["baseline_bucket", "last_bucket"]).reset_index(drop=True)

	# Draw river plot
	output_path = base_dir / "viz" / "mch_riverplot_baseline_to_last.png"
	saved = draw_two_column_river(flows, left_order, right_order, colors, output_path)

	# Print summary table
	print("Transitions (baseline_bucket -> last_bucket):")
	print(flows.to_string(index=False))
	print(f"\nSaved riverplot to: {saved}")
	
	# Longitudinal subgroup summaries (printed after saving the riverplot)
	print("\n" + "=" * 80)
	print("LONGITUDINAL SUMMARY BY BASELINE STATUS")
	print("=" * 80)
	for label in ["No MCH", "MCH"]:
		s = subgroup_longitudinal_summary(per_date, buckets, label)
		print(f"\nBaseline subgroup: {label}")
		print(f"  Patients with longitudinal MCH assessments, n(%) = {s['n_with_followup']} ({s['pct_with_followup']:.1f}%) of {s['n_total']}")
		print(f"  Number of longitudinal MCH assessments, mean ± sd = {s['mean_followup_assess']:.2f} ± {s['sd_followup_assess']:.2f}")
		print(f"  Duration of longitudinal follow-up from baseline (days), mean ± sd = {s['mean_followup_days']:.1f} ± {s['sd_followup_days']:.1f}")
		print(f"  Incidence of MCH after baseline, n(%) = {s['n_mch_after_baseline']} ({s['pct_mch_after_baseline']:.1f}%)")

	print("\n" + "=" * 80)


if __name__ == "__main__":
	user_csv = sys.argv[1] if len(sys.argv) > 1 else None
	main(user_csv)


