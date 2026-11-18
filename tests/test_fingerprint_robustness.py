from PIL import Image

from data.benchmarks.fingerprint_robustness import run_experiment


def test_fingerprint_experiment_reports_all_transformations(tmp_path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (320, 240), color=(80, 140, 200)).save(source_path)

    report = run_experiment(source_path)

    assert len(report["results"]) == 10
    for result in report["results"]:
        assert isinstance(result["sha256_matches"], bool)
        assert 0 <= result["phash_hamming_distance"] <= 64
        assert 0 <= result["dhash_hamming_distance"] <= 64
