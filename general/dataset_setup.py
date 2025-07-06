import shutil
from pathlib import Path

import kagglehub   # pip install kagglehub


def download_dataset_to_subfolder(dataset_id: str, base_dir: Path = "data") -> Path:
    """
    Download a Kaggle dataset via kagglehub and copy it into
    <project_root>/<base_dir>/<dataset_slug>/.

    :param dataset_id: e.g. "manisha717/dataset-of-pdf-files"
    :param base_dir:  top-level folder under your project (default "data")
    :return: Path to the copied dataset in your project
    """

    # Derive a folder name like "dataset-of-pdf-files"
    slug = dataset_id.split("/")[-1]
    target_dir = base_dir / slug

    # Wipe & recreate subfolder
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True)

    # Download into KaggleHub's local cache; returns that cache path
    downloaded = kagglehub.dataset_download(dataset_id)

    src = Path(downloaded)
    if not src.exists():
        raise FileNotFoundError(f"Downloaded path not found: {src}")

    # Copy files (or single file) into our target_dir
    if src.is_dir():
        for item in src.iterdir():
            dst = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
    else:  # single file
        shutil.copy2(src, target_dir / src.name)

    return target_dir
