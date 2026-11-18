from pathlib import Path


def test_celebdf_source_key_is_identity_scoped():
    from data.benchmarks.train_videomae import _source_key

    assert _source_key(Path("Celeb-real/id12_0006.mp4")) == "id12"
    assert _source_key(Path("Celeb-synthesis/id12_id20_0006.mp4")) == "id12"


def test_balanced_split_is_source_disjoint():
    from data.benchmarks.train_videomae import _split

    records = []
    for identity in range(50):
        for clip in range(2):
            real = Path(f"Celeb-real/id{identity}_{clip:04d}.mp4")
            fake = Path(f"Celeb-synthesis/id{identity}_id1_{clip:04d}.mp4")
            records.extend([(real, 0), (fake, 1)])

    training, validation = _split(records, seed=42)
    train_real = sum(label == 0 for _, label in training)
    train_fake = sum(label == 1 for _, label in training)
    val_real = sum(label == 0 for _, label in validation)
    val_fake = sum(label == 1 for _, label in validation)

    assert train_real == train_fake
    assert val_real == val_fake
    train_sources = {_source(record[0]) for record in training}
    val_sources = {_source(record[0]) for record in validation}
    assert train_sources.isdisjoint(val_sources)


def _source(path):
    from data.benchmarks.train_videomae import _source_key
    return _source_key(path)
