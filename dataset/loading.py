"""Loader construction and bounded evaluation without resetting training."""

import copy
import torch
from torch.utils.data import DataLoader
from dataset.preprocess import seed_stereo_worker


def make_loader(dataset, config, *, training=False, seed_offset=0):
    workers = config.num_workers if training else config.eval_num_workers
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    options = dict(
        batch_size=config.batch_size if training else config.batch_size_val,
        shuffle=False if training else True,
        sampler=(
            torch.utils.data.RandomSampler(
                dataset,
                generator=torch.Generator().manual_seed(config.seed + seed_offset),
            )
            if training
            else None
        ),
        num_workers=workers,
        pin_memory=torch.device(config.device).type == "cuda",
        worker_init_fn=seed_stereo_worker,
        generator=torch.Generator().manual_seed(config.seed + seed_offset),
    )
    if workers:
        options.update(
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            # multiprocessing_context="spawn",
            # timeout=getattr(config, "loader_timeout", 120),
        )
    return DataLoader(dataset, **options)


def make_train_eval_loader(train_loader, config):
    # Separate dataset AND generator: with num_workers=0, reusing the same
    # dataset would advance the training augmentation generator during eval.
    dataset = copy.deepcopy(train_loader.dataset)
    dataset.split = "train_eval"
    dataset.transformation = None
    transform = getattr(dataset, "transformation", None)
    if transform is not None:
        transform.set_random_seed(config.seed + 3)
    return make_loader(dataset, config, seed_offset=3)


def batches_for_evaluation(loader, count):
    """Evaluate at most count batches, without repeating any sample."""
    from itertools import islice

    if count is not None and count <= 0:
        raise ValueError("eval_iter must be positive or None (full evaluation)")
    if len(loader) == 0:
        raise ValueError("Evaluation loader is empty")
    return islice(loader, count)
