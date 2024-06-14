"""Main script to run dataset preprocessing pipelines."""

from config.runner_config import datasets

for dataset in datasets:
    # Check whether source_path is defined
    if not dataset.args["source_path"]:
        print(f"{dataset.name} skipped, as no source path found.")
    elif "labels_path" in dataset.args.keys() and dataset.args["labels_path"] == "":
        print(
            f"{dataset.name} skipped, as no labels path found. This dataset requires labels path to extract annotations."
        )
    elif "masks_path" in dataset.args.keys() and dataset.args["masks_path"] == "":
        print(
            f"{dataset.name} skipped, as no masks path found. This dataset requires masks path to extract annotations."
        )
    else:
        print(dataset.name)
        pipeline = dataset.pipeline
        pipeline.transform(dataset.args["source_path"])
        print(f"{dataset.name} done.")
