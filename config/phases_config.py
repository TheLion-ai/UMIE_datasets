"""Which phases are present in which datasets and how to encode them."""

phases = {
    "KITS23": {
        "0": "CT",  # Arterial or nephrogenic
    },
    "MosMedData": {
        "0": "HRCT_nocontrast",
    },
    "LIDC-IDRI": {
        "0": "CT",
    },
    "CT_COLONOGRAPHY": {
        "0": "CT",
    },
    "Chest_X-ray_Abnormalities_Detection": {
        "0": "Xray",
    },
    "CoronaHack_Chest_X-Ray_Dataset": {
        "0": "Xray",
    },
    "ChestX-ray14": {
        "0": "Xray",
    },
    "PadChest": {
        "0": "Xray",
    },
    "Lung_segmentation_from_Chest_X-Rays": {
        "0": "Xray",
    },
    "Brain_Tumor_Classification_MRI": {
        "0": "T1_weighted_postCM",  # occasionally T2_weighted! TODO: check when this dataset is implemented
    },
    "Brain_Tumor_Progression": {
        "0": "T1_weighted",  # probably some before CM and some after CM
    },
    "Qin_Brain_MRI": {
        "0": "T1_weighted_after_CM",
    },
    "BRAIN_MRI_SEGMENTATION": {
        "0": "CT",
    },
    "CT_ORG": {
        "0": "CT",
    },
    "StanfordBrainMET": {
        "0": "T1_weighted_preCM_spin-echo_pre-contrast",
        "1": "T1_weighted_postCM",  # This one was used to generate the masks
        "2": "T1_gradient_echo_postCM",  # (using an IR-prepped FSPGR sequence)
        "3": "T2_FLAIR_postCM",
    },
    "StanfordCOCA": {
        "0": "CT",
    },
    "Covid19_Detection": {
        "0": "Xray",
    },
}
