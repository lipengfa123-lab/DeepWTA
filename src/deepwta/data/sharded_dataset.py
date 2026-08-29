from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ShardedGraphDataset(Dataset):
    """Random-access dataset over shard-based PyG `.pt` files."""

    def __init__(self, dataset_dir: Path | str, split: str, cache_size: int = 4):
        self.split_dir = Path(dataset_dir) / split
        self.shard_paths = sorted(self.split_dir.glob("part_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"no shards found under {self.split_dir}")

        self.cache_size = max(int(cache_size), 1)
        self._cache: OrderedDict[int, list] = OrderedDict()

        self.shard_lengths = self._load_shard_lengths(split)

        self.cumulative: list[int] = []
        total = 0
        for length in self.shard_lengths:
            total += length
            self.cumulative.append(total)
        self.total = total

    def _load_shard_lengths(self, split: str) -> list[int]:
        """Read lengths from the dataset manifest, with a legacy scan fallback.

        Older code deserialized every shard merely to call ``len``. For the v4/v5
        datasets, the frozen manifest already records total split size and shard
        size, so startup now performs no graph deserialization.
        """
        dataset_dir = self.split_dir.parent
        data_dir = dataset_dir.parent.parent
        manifest_path = data_dir / "splits" / f"{dataset_dir.name}.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            split_config = manifest.get("split", {})
            total = int(split_config.get(split, 0))
            shard_size = int(split_config.get("shard_size", 0))
            if total > 0 and shard_size > 0:
                full, remainder = divmod(total, shard_size)
                lengths = [shard_size] * full
                if remainder:
                    lengths.append(remainder)
                if len(lengths) == len(self.shard_paths):
                    return lengths

        # Compatibility fallback for custom datasets without a frozen manifest.
        lengths: list[int] = []
        for path in self.shard_paths:
            graphs = torch.load(path, map_location="cpu", weights_only=False)
            lengths.append(len(graphs))
        return lengths

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_index: int):
        if shard_index in self._cache:
            graphs = self._cache.pop(shard_index)
            self._cache[shard_index] = graphs
            return graphs

        graphs = torch.load(
            self.shard_paths[shard_index],
            map_location="cpu",
            weights_only=False,
        )
        self._cache[shard_index] = graphs

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return graphs

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)

        shard_index = bisect_right(self.cumulative, index)
        shard_start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        local_index = index - shard_start
        return self._load_shard(shard_index)[local_index]
