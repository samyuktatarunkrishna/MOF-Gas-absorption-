import subprocess

def run(cmd: str):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    # Data pipeline
    run("python src/data/inspect_mofcsd.py")
    run("python src/data/crafted_id_list.py")
    run("python src/data/check_id_overlap.py")
    run("python src/data/build_labels_co2_298K_1bar.py")
    run("python src/data/merge_labels_into_mofcsd.py")
    run("python src/data/make_splits.py")

    # Graph pipeline
    run("python src/graphs/export_blackhole_edges_by_threshold.py")
    run("python src/graphs/labeled_coverage_by_threshold.py")
    run("python src/graphs/audit_split_connectivity.py")

    # Models
    run("python src/models/train_baselines.py")
    run("python src/models/train_blackhole_feature_baseline.py")

if __name__ == "__main__":
    main()