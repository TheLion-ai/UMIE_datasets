"""Source color values of masks for each dataset."""

dataset_masks = {
    "StanfordCOCA": {
        "coronary_artery_calcium": 20,
    },
    "StanfordBrainMET": {
        "brain_metastasis": 255,
    },
    "KITS23": {"kidney": 1, "kidney_tumor": 2, "kidney_cyst": 3},
    "CoronaHack_Chest_X-Ray_Dataset": {},
    "Brain_with_hemorrhage": {
        "hemorrhage": 255,
    },
    "Alzheimers_Dataset": {},
    "Brain_Tumor_Detection": {},
    "Covid19_Detection": {},
    "Finding_and_Measuring_Lungs_in_CT_Data": {
        "lungs": 255,
    },
    "Liver_And_Liver_Tumor": {"liver": 1, "liver_tumor": 1},
    "ChestX-ray14": {},
    "Knee_Osteoarthritis": {},
    "BrainTumorProgression": {
        "brain_tumor": 255,
    },
    "Brain_Tumor_Classification_MRI": {},
}
