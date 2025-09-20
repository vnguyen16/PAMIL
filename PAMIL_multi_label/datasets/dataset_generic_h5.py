"""HDF5-backed dataset loader for the multi-label PAMIL pipeline."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from .dataset_generic_npy import Generic_MIL_Dataset


class Generic_H5_MIL_Dataset(Generic_MIL_Dataset):
    """Load multi-label MIL bags stored as ``.h5`` feature files."""

    def __init__(
        self,
        data_dir: str,
        data_mag: Optional[str] = None,
        *,
        feature_key: Sequence[str] | str = ("feature2", "features", "feature"),
        coord_key: Sequence[str] | str = ("coords", "index"),
        inst_label_key: Optional[Sequence[str] | str] = ("inst_label",),
        file_suffix: str = "",
        file_ext: str = ".h5",
        use_float32: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(data_dir=data_dir, data_mag=data_mag, **kwargs)

        self._configure_h5_access(
            feature_key=feature_key,
            coord_key=coord_key,
            inst_label_key=inst_label_key,
            file_suffix=file_suffix,
            file_ext=file_ext,
            use_float32=use_float32,
        )

    # Shared helpers -----------------------------------------------------
    def _configure_h5_access(
        self,
        *,
        feature_key: Sequence[str] | str,
        coord_key: Sequence[str] | str,
        inst_label_key: Optional[Sequence[str] | str],
        file_suffix: str,
        file_ext: str,
        use_float32: bool,
    ) -> None:
        self.feature_keys = self._coerce_keys(feature_key, default=("feature2", "features", "feature"))
        self.coord_keys = self._coerce_keys(coord_key, default=("coords", "index"))
        self.inst_label_keys = self._coerce_keys(
            inst_label_key,
            default=("inst_label",),
            allow_empty=True,
        )
        self.file_suffix = file_suffix or ""
        self.file_ext = self._normalise_extension(file_ext)
        self.use_float32 = use_float32

    @staticmethod
    def _coerce_keys(
        keys: Optional[Sequence[str] | str],
        *,
        default: Optional[Iterable[str]] = None,
        allow_empty: bool = False,
    ) -> Tuple[str, ...]:
        if keys is None:
            keys = default

        if keys is None:
            return tuple() if allow_empty else ()

        if isinstance(keys, str):
            keys = (keys,)
        elif isinstance(keys, Iterable):
            keys = tuple(keys)
        else:
            raise TypeError("Keys must be a string or an iterable of strings.")

        if not keys and not allow_empty:
            raise ValueError("At least one key must be provided.")

        return keys

    @staticmethod
    def _normalise_extension(file_ext: str) -> str:
        if not file_ext:
            file_ext = ".h5"
        elif not file_ext.startswith("."):
            file_ext = f".{file_ext}"
        return file_ext

    # Core loading -------------------------------------------------------
    def _resolve_slide_path(self, slide_id: str) -> str:
        base_name = slide_id
        if self.data_mag:
            base_name = f"{base_name}_{self.data_mag}"
        if self.file_suffix:
            base_name = f"{base_name}{self.file_suffix}"
        return os.path.join(self.data_dir, f"{base_name}{self.file_ext}")

    def _select_dataset(
        self,
        handle: h5py.File,
        keys: Tuple[str, ...],
        *,
        required: bool,
        default,
    ):
        for key in keys:
            if key and key in handle:
                return handle[key][:]
        if required:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"None of the keys {keys} were found in {handle.filename}. "
                f"Available datasets: [{available}]"
            )
        return default() if callable(default) else default

    def __getitem__(self, idx: int):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        label = self.translabel(label)

        h5_path = self._resolve_slide_path(slide_id)
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Slide features not found: {h5_path}")

        with h5py.File(h5_path, 'r') as handle:
            features = self._select_dataset(handle, self.feature_keys, required=True, default=None)
            coords = self._select_dataset(
                handle,
                self.coord_keys,
                required=False,
                default=lambda: np.empty((0, 2), dtype=np.int32),
            )
            if self.inst_label_keys:
                inst_label = self._select_dataset(
                    handle,
                    self.inst_label_keys,
                    required=False,
                    default=lambda: np.array([], dtype=np.int64),
                )
            else:
                inst_label = np.array([], dtype=np.int64)

        features = torch.from_numpy(features)
        if self.use_float32 and features.dtype != torch.float32:
            features = features.float()

        coords = np.asarray(coords)
        inst_label = np.asarray(inst_label)

        return features, label, coords, inst_label, slide_id

    # Split handling -----------------------------------------------------
    def return_splits(self, from_id: bool = True, csv_path: Optional[str] = None):
        base_splits = super().return_splits(from_id=from_id, csv_path=csv_path)

        def _convert(split):
            if split is None:
                return None
            return Generic_H5_Split(
                split.slide_data,
                data_dir=self.data_dir,
                data_mag=self.data_mag,
                num_classes=self.num_classes,
                feature_key=self.feature_keys,
                coord_key=self.coord_keys,
                inst_label_key=self.inst_label_keys,
                file_suffix=self.file_suffix,
                file_ext=self.file_ext,
                use_float32=self.use_float32,
            )

        return tuple(_convert(split) for split in base_splits)


class Generic_H5_Split(Generic_H5_MIL_Dataset):
    """Split wrapper preserving the HDF5 loading behaviour."""

    def __init__(
        self,
        slide_data,
        *,
        data_dir: Optional[str],
        data_mag: Optional[str],
        num_classes: int,
        feature_key: Tuple[str, ...],
        coord_key: Tuple[str, ...],
        inst_label_key: Tuple[str, ...],
        file_suffix: str,
        file_ext: str,
        use_float32: bool,
    ) -> None:
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.data_mag = data_mag
        self.num_classes = num_classes
        self._configure_h5_access(
            feature_key=feature_key,
            coord_key=coord_key,
            inst_label_key=inst_label_key,
            file_suffix=file_suffix,
            file_ext=file_ext,
            use_float32=use_float32,
        )
        self.slide_cls_ids = [np.where(self.slide_data['label'] == i)[0] for i in range(self.num_classes)]

    def __len__(self) -> int:
        return len(self.slide_data)
