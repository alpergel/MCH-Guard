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


def bucket_label(mch_pos_flag: int, mch_count: int) -> str:
	"""Map a per-date record to a bucket label using MCH_pos_flag.

	Buckets:
	- No MCH
	- MCH=1
	- MCH=2-4
	- MCH>4
	"""
	if mch_pos_flag == 0:
		return "No MCH"
	if mch_count == 1:
		return "MCH=1"
	if 2 <= mch_count <= 4:
		return "MCH=2-4"
	if mch_count >= 5:
		return "MCH>4"
	# Fallback when positive but count is 0
	return "No MCH"


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
	# Layout parameters
	left_x = 0.05
	right_x = 0.95
	curve_x_offset = 0.35
	bar_width = 0.04
	fig_w, fig_h, dpi = 10, 8, 150

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
		ax.text(left_x - 0.07, (y0 + y1) / 2, f"{label}\n(n={left_totals.get(label, 0)})", va="center", ha="right", fontsize=10)

	for label in right_order:
		(y0, y1) = right_pos[label]
		ax.add_patch(Rectangle((right_x - bar_width / 2, y0), bar_width, y1 - y0, color="#E0E0E0"))
		ax.text(right_x + 0.07, (y0 + y1) / 2, f"{label}\n(n={right_totals.get(label, 0)})", va="center", ha="left", fontsize=10)

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
	ax.text(0.5, 1.02, "Baseline to Last MCH count categories (riverplot)", ha="center", va="bottom", fontsize=12)
	ax.text(left_x, 1.0, "Baseline", ha="center", va="bottom", fontsize=11)
	ax.text(right_x, 1.0, "Last", ha="center", va="bottom", fontsize=11)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
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


def analyze_mch_positive_baseline(per_date: pd.DataFrame, buckets: pd.DataFrame) -> Dict:
	"""Analyze MCH-positive-at-baseline patients for follow-up and outcome characteristics.

	Returns a dict with counts/percentages of follow-up, resolution, improvement,
	worsening, stability, last-bucket distribution, and follow-up duration stats.
	"""
	# Select MCH-positive at baseline (exclude "No MCH")
	mch_pos_baseline = buckets[buckets["baseline_bucket"] != "No MCH"].copy()
	total_mch_pos = len(mch_pos_baseline)

	# Follow-up subset
	with_followup = mch_pos_baseline[~mch_pos_baseline["single_visit"]].copy()
	n_with_followup = len(with_followup)
	pct_with_followup = (n_with_followup / total_mch_pos * 100) if total_mch_pos > 0 else 0.0

	# Resolution among those with follow-up
	resolved = with_followup[with_followup["last_bucket"] == "No MCH"]
	resolved_count = len(resolved)
	pct_resolved = (resolved_count / n_with_followup * 100) if n_with_followup > 0 else 0.0

	# Improvement/Worsening/Stable relative to baseline severity
	severity = {"No MCH": 0, "MCH=1": 1, "MCH=2-4": 2, "MCH>4": 3, "NODATA": -1}
	with_followup = with_followup.assign(
		baseline_sev=with_followup["baseline_bucket"].map(severity),
		last_sev=with_followup["last_bucket"].map(severity),
	)
	improved_count = int((with_followup["last_sev"] < with_followup["baseline_sev"]).sum())
	worsened_count = int((with_followup["last_sev"] > with_followup["baseline_sev"]).sum())
	stable_count = int((with_followup["last_sev"] == with_followup["baseline_sev"]).sum())
	pct_improved = (improved_count / n_with_followup * 100) if n_with_followup > 0 else 0.0
	pct_worsened = (worsened_count / n_with_followup * 100) if n_with_followup > 0 else 0.0
	pct_stable = (stable_count / n_with_followup * 100) if n_with_followup > 0 else 0.0

	# Last-bucket distribution among those with follow-up
	order = ["No MCH", "MCH=1", "MCH=2-4", "MCH>4"]
	last_counts = with_followup["last_bucket"].value_counts(dropna=False)
	last_bucket_counts = {label: int(last_counts.get(label, 0)) for label in order}
	last_bucket_percents = {label: ((last_bucket_counts[label] / n_with_followup) * 100 if n_with_followup > 0 else 0.0) for label in order}

	# Baseline composition within positive group
	base_counts = mch_pos_baseline["baseline_bucket"].value_counts()
	baseline_bucket_counts = {label: int(base_counts.get(label, 0)) for label in ["MCH=1", "MCH=2-4", "MCH>4"]}

	# Follow-up duration (years)
	per_date_sorted = per_date.sort_values(["RID", "SCANDATE"]).copy()
	baseline_dates = per_date_sorted.groupby("RID", as_index=False).first()[["RID", "SCANDATE"]].rename(columns={"SCANDATE": "baseline_date"})
	last_dates = per_date_sorted.groupby("RID", as_index=False).last()[["RID", "SCANDATE"]].rename(columns={"SCANDATE": "last_date"})
	dates_merged = baseline_dates.merge(last_dates, on="RID")
	dates_merged["baseline_date"] = pd.to_datetime(dates_merged["baseline_date"])
	dates_merged["last_date"] = pd.to_datetime(dates_merged["last_date"])
	dates_merged["followup_days"] = (dates_merged["last_date"] - dates_merged["baseline_date"]).dt.days
	dates_merged["followup_years"] = dates_merged["followup_days"] / 365.25
	with_followup_dates = with_followup.merge(dates_merged, on="RID", how="left")
	followup_years = with_followup_dates["followup_years"].dropna()
	followup_years_mean = float(followup_years.mean()) if len(followup_years) > 0 else 0.0
	followup_years_std = float(followup_years.std()) if len(followup_years) > 0 else 0.0

	return {
		"total_mch_pos_baseline": total_mch_pos,
		"n_with_followup": n_with_followup,
		"pct_with_followup": pct_with_followup,
		"resolved_count": resolved_count,
		"pct_resolved": pct_resolved,
		"improved_count": improved_count,
		"pct_improved": pct_improved,
		"worsened_count": worsened_count,
		"pct_worsened": pct_worsened,
		"stable_count": stable_count,
		"pct_stable": pct_stable,
		"last_bucket_counts": last_bucket_counts,
		"last_bucket_percents": last_bucket_percents,
		"baseline_bucket_counts": baseline_bucket_counts,
		"followup_years_mean": followup_years_mean,
		"followup_years_std": followup_years_std,
	}


def summarize_mch_pos_vs_neg_baseline(df: pd.DataFrame, buckets: pd.DataFrame) -> Dict[str, Dict[str, float]]:
	"""Compare baseline characteristics between MCH-positive and MCH-negative participants.

	Computes group means/SDs and p-values (Welch's t-test) for continuous vars,
	and proportions with p-values (chi-square/Fisher) for binary vars.
	"""
	# Safe import of scipy; allow fallback without p-values if unavailable
	try:
		from scipy import stats as scipy_stats  # type: ignore
	except Exception:  # pragma: no cover
		scipy_stats = None  # type: ignore

	# Build baseline cohort rows (earliest SCANDATE per RID)
	baseline_rows = (
		df.sort_values(["RID", "SCANDATE"]).groupby("RID", as_index=False).first()
	)
	# Attach baseline MCH bucket
	baseline_with_bucket = baseline_rows.merge(
		buckets[["RID", "baseline_bucket"]], on="RID", how="left"
	)
	# Define groups
	pos_group = baseline_with_bucket[baseline_with_bucket["baseline_bucket"] != "No MCH"].copy()
	neg_group = baseline_with_bucket[baseline_with_bucket["baseline_bucket"] == "No MCH"].copy()

	def summarize_continuous(series_pos: pd.Series, series_neg: pd.Series) -> Tuple[float, float, float, float, float]:
		pos_clean = pd.to_numeric(series_pos, errors="coerce").dropna()
		neg_clean = pd.to_numeric(series_neg, errors="coerce").dropna()
		pos_mean = float(pos_clean.mean()) if len(pos_clean) else float("nan")
		pos_sd = float(pos_clean.std(ddof=1)) if len(pos_clean) > 1 else float("nan")
		neg_mean = float(neg_clean.mean()) if len(neg_clean) else float("nan")
		neg_sd = float(neg_clean.std(ddof=1)) if len(neg_clean) > 1 else float("nan")
		if scipy_stats is not None and len(pos_clean) > 1 and len(neg_clean) > 1:
			_, p_val = scipy_stats.ttest_ind(pos_clean, neg_clean, equal_var=False, nan_policy="omit")
			p_val = float(p_val) if p_val is not None else float("nan")
		else:
			p_val = float("nan")
		return pos_mean, pos_sd, neg_mean, neg_sd, p_val

	def to_bool(series: pd.Series) -> pd.Series:
		if series.dtype == bool:
			return series
		# Treat non-zero numeric as True; for strings, look for affirmative markers
		if pd.api.types.is_numeric_dtype(series):
			return series.fillna(0).astype(float) > 0
		lower = series.astype(str).str.lower()
		return lower.isin(["1", "true", "yes", "y", "male", "m", "pos", "positive"]) | (lower == "t")

	def summarize_binary(series_pos: pd.Series, series_neg: pd.Series) -> Tuple[float, float]:
		pos_bool = to_bool(series_pos)
		neg_bool = to_bool(series_neg)
		pos_rate = float(pos_bool.mean() * 100.0)
		neg_rate = float(neg_bool.mean() * 100.0)
		# 2x2 table for p-value
		a = int(pos_bool.sum())
		b = int((~pos_bool).sum())
		c = int(neg_bool.sum())
		d = int((~neg_bool).sum())
		p_val = float("nan")
		if scipy_stats is not None:
			try:
				chi2, p_val, _, _ = scipy_stats.chi2_contingency([[a, b], [c, d]], correction=True)
				p_val = float(p_val)
			except Exception:
				try:
					_, p_val = scipy_stats.fisher_exact([[a, b], [c, d]])
					p_val = float(p_val)
				except Exception:
					p_val = float("nan")
		return pos_rate, neg_rate, p_val

	results: Dict[str, Dict[str, float]] = {}

	# Continuous features
	for col in [
		"PTAGE",  # Age
		"PHC_MEM",
		"PHC_EXF",
		"PHC_LAN",
		"CDRSB",
		"ptau_ab_ratio_csf",
		"PLASMA_NFL",
		"PHC_BMI",
	]:
		if col in baseline_with_bucket.columns:
			pos_mean, pos_sd, neg_mean, neg_sd, p_val = summarize_continuous(pos_group[col], neg_group[col])
			results[col] = {
				"pos_mean": pos_mean,
				"pos_sd": pos_sd,
				"neg_mean": neg_mean,
				"neg_sd": neg_sd,
				"p": p_val,
			}

	# Binary features
	# Sex (Male): PTGENDER coded as 1=Male, 2=Female (robust to string/numeric)
	if "PTGENDER" in baseline_with_bucket.columns:
		def male_mask(series: pd.Series) -> pd.Series:
			num = pd.to_numeric(series, errors="coerce")
			if num.notna().any():
				return num == 1
			lower = series.astype(str).str.lower()
			return (lower == "1") | lower.str.startswith("m") | (lower == "male")
		male_pos = male_mask(pos_group["PTGENDER"])
		male_neg = male_mask(neg_group["PTGENDER"])
		pos_rate, neg_rate, p_val = summarize_binary(male_pos, male_neg)
		results["Male"] = {"pos_pct": pos_rate, "neg_pct": neg_rate, "p": p_val}

	# APOE e4 homozygotes: e4_GENOTYPE == 2
	if "e4_GENOTYPE" in baseline_with_bucket.columns:
		pos_rate, neg_rate, p_val = summarize_binary(pos_group["e4_GENOTYPE"] == 2, neg_group["e4_GENOTYPE"] == 2)
		results["APOE_e4_homozygote"] = {"pos_pct": pos_rate, "neg_pct": neg_rate, "p": p_val}

	# Cardiac and Renal conditions (domain flags)
	for domain_col, label in [("CARD", "Cardiac"), ("RENA", "Renal")]:
		if domain_col in baseline_with_bucket.columns:
			pos_rate, neg_rate, p_val = summarize_binary(pos_group[domain_col], neg_group[domain_col])
			results[f"Domain_{label}"] = {"pos_pct": pos_rate, "neg_pct": neg_rate, "p": p_val}

	# AD/Dementia-related medications
	if "MED_AD_and_Dementia" in baseline_with_bucket.columns:
		pos_rate, neg_rate, p_val = summarize_binary(pos_group["MED_AD_and_Dementia"], neg_group["MED_AD_and_Dementia"])
		results["Med_AD_and_Dementia"] = {"pos_pct": pos_rate, "neg_pct": neg_rate, "p": p_val}

	# Group sizes
	results["_meta"] = {
		"n_pos": float(len(pos_group)),
		"n_neg": float(len(neg_group)),
	}
	return results


def main(csv_path: str | None = None) -> None:
	base_dir = Path(__file__).resolve().parents[2]
	default_csv = base_dir / "processed" / "merge.csv"
	csv_file = Path(csv_path) if csv_path else default_csv

	df = load_dataset(csv_file)
	per_date = compute_per_date_counts(df)
	buckets = baseline_last_buckets(per_date)
	flows = flow_counts(buckets)

	# Define category orders and colors
	left_order = [
		"No MCH",
		"MCH=1",
		"MCH=2-4",
		"MCH>4",
	]
	right_order = [
		"No MCH",
		"MCH=1",
		"MCH=2-4",
		"MCH>4",
		"NODATA",
	]
	colors = {
		"No MCH": "#6BA292",
		"MCH=1": "#5B84B1",
		"MCH=2-4": "#F6BD60",
		"MCH>4": "#EE6C4D",
	}

	# Ensure flows include only categories present in orders; coerce unknowns to 'MCH=0 (flag=0)' if needed
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
	
	# Analyze MCH-negative baseline progression
	print("\n" + "=" * 80)
	print("MCH-NEGATIVE BASELINE ANALYSIS")
	print("=" * 80)
	
	stats = analyze_mch_negative_progression(per_date, buckets)
	
	print(f"\nTotal MCH-negative at baseline: {stats['total_mch_neg_baseline']}")
	print(f"\nAmong those who were MCH-negative at baseline:")
	print(f"  - {stats['pct_with_followup']:.1f}% had follow-up MCH assessment data (n={stats['n_with_followup']})")
	print(f"  - {stats['pct_developed_mch']:.1f}% developed MCH during follow-up (n={stats['n_developed_mch']} of {stats['n_with_followup']} with follow-up)")
	print(f"  - Follow-up duration: {stats['followup_years_mean']:.2f} ± {stats['followup_years_std']:.2f} years")
	
	print("\n" + "=" * 80)
	print("MCH-POSITIVE BASELINE ANALYSIS")
	print("=" * 80)

	pos_stats = analyze_mch_positive_baseline(per_date, buckets)

	print(f"\nTotal MCH-positive at baseline: {pos_stats['total_mch_pos_baseline']}")
	print(f"\nAmong those who were MCH-positive at baseline:")
	print(f"  - {pos_stats['pct_with_followup']:.1f}% had follow-up MCH assessment data (n={pos_stats['n_with_followup']})")
	print(f"  - Resolution to No MCH at last follow-up: {pos_stats['pct_resolved']:.1f}% (n={pos_stats['resolved_count']})")
	print(f"  - Improved vs baseline: {pos_stats['pct_improved']:.1f}% (n={pos_stats['improved_count']})")
	print(f"  - Worsened vs baseline: {pos_stats['pct_worsened']:.1f}% (n={pos_stats['worsened_count']})")
	print(f"  - Stable vs baseline: {pos_stats['pct_stable']:.1f}% (n={pos_stats['stable_count']})")
	print("  - Last-bucket distribution among those with follow-up:")
	print(f"      No MCH: {pos_stats['last_bucket_counts']['No MCH']} ({pos_stats['last_bucket_percents']['No MCH']:.1f}%)")
	print(f"      MCH=1: {pos_stats['last_bucket_counts']['MCH=1']} ({pos_stats['last_bucket_percents']['MCH=1']:.1f}%)")
	print(f"      MCH=2-4: {pos_stats['last_bucket_counts']['MCH=2-4']} ({pos_stats['last_bucket_percents']['MCH=2-4']:.1f}%)")
	print(f"      MCH>4: {pos_stats['last_bucket_counts']['MCH>4']} ({pos_stats['last_bucket_percents']['MCH>4']:.1f}%)")
	print(f"  - Follow-up duration: {pos_stats['followup_years_mean']:.2f} ± {pos_stats['followup_years_std']:.2f} years")
	print("  - Baseline composition (within MCH-positive):")
	print(f"      MCH=1: {pos_stats['baseline_bucket_counts']['MCH=1']}")
	print(f"      MCH=2-4: {pos_stats['baseline_bucket_counts']['MCH=2-4']}")
	print(f"      MCH>4: {pos_stats['baseline_bucket_counts']['MCH>4']}")

	print("\n" + "=" * 80)
	print("MCH-POSITIVE VS MCH-NEGATIVE BASELINE CHARACTERISTICS")
	print("=" * 80)

	char = summarize_mch_pos_vs_neg_baseline(df, buckets)
	pos_n = int(char["_meta"]["n_pos"]) if "_meta" in char else 0
	neg_n = int(char["_meta"]["n_neg"]) if "_meta" in char else 0

	def fmt_mean_sd(key: str, label: str) -> None:
		if key in char:
			print(
				f"  - {label}: {char[key]['pos_mean']:.2f} ± {char[key]['pos_sd']:.2f} (MCH-pos, n={pos_n}) vs "
				f"{char[key]['neg_mean']:.2f} ± {char[key]['neg_sd']:.2f} (MCH-neg, n={neg_n}); p={char[key]['p']:.3g}"
			)

	def fmt_prop(key: str, label: str) -> None:
		if key in char:
			print(
				f"  - {label}: {char[key]['pos_pct']:.1f}% (MCH-pos, n={pos_n}) vs {char[key]['neg_pct']:.1f}% (MCH-neg, n={neg_n}); p={char[key]['p']:.3g}"
			)

	# Demographics
	fmt_mean_sd("PTAGE", "Age (years)")
	fmt_prop("Male", "Male sex")
	fmt_prop("APOE_e4_homozygote", "APOE ε4 homozygotes")

	# Cognitive performance / severity
	fmt_mean_sd("PHC_MEM", "Memory score")
	fmt_mean_sd("PHC_EXF", "Executive function score")
	fmt_mean_sd("PHC_LAN", "Language score")
	fmt_mean_sd("CDRSB", "CDR-SB")

	# Biomarkers
	fmt_mean_sd("ptau_ab_ratio_csf", "CSF p-tau/Aβ42 ratio")
	fmt_mean_sd("PLASMA_NFL", "Plasma NfL")

	# Health/clinical
	fmt_mean_sd("PHC_BMI", "BMI")
	fmt_prop("Domain_Cardiac", "Cardiac conditions")
	fmt_prop("Domain_Renal", "Renal conditions")

	# Medications
	fmt_prop("Med_AD_and_Dementia", "AD & dementia-related medications")

	print("\n" + "=" * 80)


if __name__ == "__main__":
	user_csv = sys.argv[1] if len(sys.argv) > 1 else None
	main(user_csv)


