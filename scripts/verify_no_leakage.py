"""Verify no data leakage in generated splits."""

import json
from collections import defaultdict
from pathlib import Path


def check_experiment_leakage(exp_dir: Path) -> dict:
    """Check for data leakage in a single experiment."""
    test_file = exp_dir / "test.json"

    if not test_file.exists():
        return {"error": "Missing test.json"}

    # Determine if this is SimpleMixed or FineTune strategy
    exp_name = exp_dir.name
    is_finetune = exp_name.startswith("finetune_")

    with open(test_file) as f:
        test_files = set(json.load(f))

    # For FT strategy, check both train_synthetic.json and train_real.json
    # For SM strategy, check train.json
    if is_finetune:
        train_synthetic_file = exp_dir / "train_synthetic.json"
        train_real_file = exp_dir / "train_real.json"

        if not train_synthetic_file.exists() or not train_real_file.exists():
            return {"error": "Missing train_synthetic.json or train_real.json"}

        with open(train_synthetic_file) as f:
            synthetic_files = set(json.load(f))
        with open(train_real_file) as f:
            real_files = set(json.load(f))

        overlap_synthetic = synthetic_files & test_files
        overlap_real = real_files & test_files
        total_train_files = synthetic_files | real_files

        return {
            "strategy": "FineTune (FT)",
            "train_files": ["train_synthetic.json", "train_real.json"],
            "synthetic_count": len(synthetic_files),
            "real_count": len(real_files),
            "test_count": len(test_files),
            "overlap_synthetic_count": len(overlap_synthetic),
            "overlap_real_count": len(overlap_real),
            "overlap_total_count": len(total_train_files & test_files),
            "overlap_files": list(total_train_files & test_files)
            if (total_train_files & test_files)
            else [],
        }
    else:
        train_file = exp_dir / "train.json"

        if not train_file.exists():
            return {"error": "Missing train.json"}

        with open(train_file) as f:
            train_files = set(json.load(f))

        overlap = train_files & test_files

        return {
            "strategy": "SimpleMixed (SM)",
            "train_files": ["train.json"],
            "train_count": len(train_files),
            "test_count": len(test_files),
            "overlap_count": len(overlap),
            "overlap_files": list(overlap) if overlap else [],
        }


def main():
    base_dir = Path("data/model_ready/wachter_splits")

    # Define all experiment directories
    experiment_dirs = [
        "sm_box",
        "ft_box",
        "sm_slope",
        "ft_slope",
        "sm_gen_pattern_box",
        "ft_gen_pattern_box",
        "sm_pattern_box",
        "ft_pattern_box",
        "sm_gen_cardboard_box",
        "ft_gen_cardboard_box",
        "sm_cardboard_box",
        "ft_cardboard_box",
        "sm_gen_larvik",
        "ft_gen_larvik",
        "sm_larvik",
        "ft_larvik",
        "sm_gen_rv4",
        "ft_gen_rv4",
        "sm_rv4",
        "ft_rv4",
    ]

    all_results = {}
    leakage_found = []

    print("Checking all 140 experiments for data leakage...")
    print("=" * 80)

    for exp_type in experiment_dirs:
        exp_type_dir = base_dir / exp_type
        if not exp_type_dir.exists():
            print(f"[SKIP] {exp_type}: directory not found")
            continue

        # Check each experiment configuration in this type
        for exp_config_dir in sorted(exp_type_dir.iterdir()):
            if not exp_config_dir.is_dir():
                continue

            result = check_experiment_leakage(exp_config_dir)
            exp_name = f"{exp_type}/{exp_config_dir.name}"
            all_results[exp_name] = result

            # Check for leakage based on strategy
            overlap_count = result.get(
                "overlap_total_count", result.get("overlap_count", 0)
            )

            if overlap_count > 0:
                leakage_found.append(exp_name)
                strategy = result.get("strategy", "Unknown")
                train_files = ", ".join(result.get("train_files", []))

                print(f"[LEAK] {exp_name} ({strategy})")
                print(f"       Train files: {train_files}")

                if "synthetic_count" in result:
                    print(
                        f"       Synthetic: {result['synthetic_count']}, Real: {result['real_count']}, Test: {result['test_count']}"
                    )
                    if result["overlap_synthetic_count"] > 0:
                        print(
                            f"       train_synthetic.json ∩ test.json: {result['overlap_synthetic_count']} files"
                        )
                    if result["overlap_real_count"] > 0:
                        print(
                            f"       train_real.json ∩ test.json: {result['overlap_real_count']} files"
                        )
                else:
                    print(
                        f"       Train: {result['train_count']}, Test: {result['test_count']}"
                    )

                print(f"       Total overlap: {overlap_count} files")
                print(f"       Sample files: {result['overlap_files'][:3]}...")
            elif "error" in result:
                print(f"[ERROR] {exp_name}: {result['error']}")

    print("=" * 80)
    print(f"\nTotal experiments checked: {len(all_results)}")

    if leakage_found:
        print(f"\n[CRITICAL] Data leakage found in {len(leakage_found)} experiments:")
        for exp in leakage_found:
            overlap = all_results[exp].get(
                "overlap_total_count", all_results[exp].get("overlap_count", 0)
            )
            strategy = all_results[exp].get("strategy", "Unknown")
            print(f"  - {exp} ({strategy}): {overlap} overlapping files")
    else:
        print("\n[SUCCESS] No data leakage detected in any experiment!")

    # Check test set consistency within each experiment type
    print("\n" + "=" * 80)
    print("Checking test set consistency within experiment types...")
    print("=" * 80)

    test_set_consistency = defaultdict(list)

    for exp_name, result in all_results.items():
        if "error" not in result:
            exp_type = exp_name.split("/")[0]
            test_file = base_dir / exp_name / "test.json"
            with open(test_file) as f:
                test_files = tuple(sorted(json.load(f)))
            test_set_consistency[exp_type].append((exp_name, test_files))

    inconsistent_types = []
    for exp_type, experiments in test_set_consistency.items():
        if len(experiments) > 1:
            # Check if all test sets are the same
            first_test_set = experiments[0][1]
            all_same = all(test_set == first_test_set for _, test_set in experiments)

            if all_same:
                print(
                    f"[OK] {exp_type}: All {len(experiments)} experiments use same test set ({len(first_test_set)} files)"
                )
            else:
                inconsistent_types.append(exp_type)
                print(f"[WARN] {exp_type}: Inconsistent test sets across experiments")
                for exp_name, test_set in experiments[:3]:  # Show first 3
                    print(f"       {exp_name}: {len(test_set)} files")

    print("=" * 80)
    if inconsistent_types:
        print(
            f"\n[WARNING] {len(inconsistent_types)} experiment types have inconsistent test sets"
        )
    else:
        print("\n[SUCCESS] All experiment types have consistent test sets!")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)

    for exp_type in experiment_dirs:
        type_results = {
            k: v for k, v in all_results.items() if k.startswith(exp_type + "/")
        }
        if type_results:
            total = len(type_results)
            strategy = list(type_results.values())[0].get("strategy", "Unknown")

            if "synthetic_count" in list(type_results.values())[0]:
                avg_synth = (
                    sum(r.get("synthetic_count", 0) for r in type_results.values())
                    / total
                )
                avg_real = (
                    sum(r.get("real_count", 0) for r in type_results.values()) / total
                )
                avg_test = (
                    sum(r.get("test_count", 0) for r in type_results.values()) / total
                )
                print(
                    f"{exp_type:30s}: {total:2d} configs ({strategy}), avg synth={avg_synth:6.1f}, real={avg_real:6.1f}, test={avg_test:5.1f}"
                )
            else:
                avg_train = (
                    sum(r.get("train_count", 0) for r in type_results.values()) / total
                )
                avg_test = (
                    sum(r.get("test_count", 0) for r in type_results.values()) / total
                )
                print(
                    f"{exp_type:30s}: {total:2d} configs ({strategy}), avg train={avg_train:6.1f}, test={avg_test:5.1f}"
                )


if __name__ == "__main__":
    main()
