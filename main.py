"""Docstring."""
import yaml

if __name__ == "__main__":
    """Docstring."""
    datasets = yaml.load(open("config/runner_config.yaml"), Loader=yaml.FullLoader)

    #  Automatically import and run function {function_name} if source_path is defined
    #  in config/runner_config.yaml file.
    for dataset in datasets.keys():

        # Check whether source_path is defined
        if not datasets[dataset]["args"]["source_path"]:
            print(f"{dataset} skipped, as no source path found.")
        else:
            print(dataset)

            # List arguments
            args = datasets[dataset]["args"]
            for argument in args.keys():
                print(f"{argument}: {args[argument]}")

            # Import preprocessing function
            exec(datasets[dataset]["import_statement"])

            # Run preprocessing function
            exec(f"{datasets[dataset]['function_name']}(**args)")
