"""TODO: A script counting the number of data in the database with specific masks and labels."""
import json

# Path to the JSON file
file_path = "testing/test_dummy_data/00_KITS23/input/kits23.json"

# Set to store unique tumor histologic subtypes
unique_subtypes = set()

# Open and load the JSON data
with open(file_path, "r") as file:
    data = json.load(file)
    # Iterate through each case in the JSON data
    for case in data:
        # Extract the tumor_histologic_subtype and add to the set
        subtype = case.get("tumor_histologic_subtype", None)
        if subtype:
            unique_subtypes.add(subtype)

# Print the unique tumor histologic subtypes
print("Unique Tumor Histologic Subtypes:")
for subtype in unique_subtypes:
    print(subtype)
