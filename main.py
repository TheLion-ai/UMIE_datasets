"""Main script to run dataset preprocessing pipelines."""

from config.runner_config import datasets

for dataset in datasets:
    # Check whether source_path is defined
    if not dataset.args["source_path"]:
        print(f"{dataset.name} skipped, as no source path found.")
    else:
        print(dataset.name)
        pipeline = dataset.pipeline
        pipeline.transform(dataset.args["source_path"])
        print(f"{dataset.name} done.")
