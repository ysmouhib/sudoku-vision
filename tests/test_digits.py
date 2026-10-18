import torch

from digits import cell_dataset, digit_pool, make_blanks, make_model, split_indices


def test_split_is_a_disjoint_90_10_partition():
    train, val = split_indices(1000)
    assert len(train) == 900 and len(val) == 100
    assert sorted(torch.cat([train, val]).tolist()) == list(range(1000))


def test_split_is_fixed_by_the_seed():
    assert torch.equal(split_indices(500)[1], split_indices(500)[1])


def test_validation_digits_are_not_training_digits():
    train_images, _ = digit_pool("train")
    val_images, _ = digit_pool("val")
    train_set = {img.numpy().tobytes() for img in train_images[:, 0]}
    shared = sum(img.numpy().tobytes() in train_set for img in val_images[:, 0])
    assert shared <= 5  # MNIST has a few exact duplicate images; a leak would give thousands


def test_no_mnist_zero_is_left():
    for split in ("train", "val", "test"):
        assert (digit_pool(split)[1] != 0).all()


def test_blanks_are_dark_noise_in_range():
    blanks = make_blanks(100, seed=0)
    assert blanks.shape == (100, 1, 28, 28)
    assert blanks.min() >= 0 and blanks.max() <= 1
    assert blanks.mean() < 0.05


def test_blanks_are_fixed_by_the_seed():
    assert torch.equal(make_blanks(3, seed=7), make_blanks(3, seed=7))
    assert not torch.equal(make_blanks(3, seed=7), make_blanks(3, seed=8))


def test_cell_dataset_has_about_one_class_of_blanks():
    images, labels = cell_dataset("val")
    counts = torch.bincount(labels, minlength=10)
    assert len(images) == len(labels)
    assert abs(counts[0].item() - counts[1:].float().mean().item()) < 0.1 * counts[0].item()


def test_models_output_ten_logits():
    x = torch.rand(4, 1, 28, 28)
    for name in ("linear", "mlp", "cnn"):
        assert make_model(name)(x).shape == (4, 10)
