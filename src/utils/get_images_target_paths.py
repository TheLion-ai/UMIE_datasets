"""Get listing images paths."""
import glob
import os


def get_images_paths(
    phases: dict,
    target_path: str,
    dataset_uid: str,
    dataset_name: str,
    image_folder_name: str,
) -> list:
    """List paths to all images based on parameters."""
    new_paths = []
    if len(phases.keys()) <= 1:
        root_path = os.path.join(
            target_path,
            f"{dataset_uid}_{dataset_name}",
            image_folder_name,
        )
        new_paths = glob.glob(f"{root_path}/**/*.png", recursive=True)
    else:
        for phase in phases.values():
            root_path = os.path.join(
                target_path,
                f"{dataset_uid}_{dataset_name}",
                phase,
                image_folder_name,
            )
            new_paths.extend(glob.glob(f"{root_path}/**/*.png", recursive=True))
    return new_paths
