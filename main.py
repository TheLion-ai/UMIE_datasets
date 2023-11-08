from pipelines.ct.stanford_coca import preprocess_coca
from pipelines.mri.stanford_brain_met import preprocess_stanford_brain_met
from pipelines.ct.kits19 import preprocess_kits19

# preprocess_coca(
#     source_path="/home/basia/Desktop/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/patient",
#     target_path="./data/",
#     masks_path="/home/basia/Desktop/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/calcium_xml"
#

# preprocess_stanford_brain_met(
#     # We use only train set since test set has no masks
#     source_path='/home/basia/Desktop/BrainMetShare/stanford_release_brainmask/mets_stanford_releaseMask_train',
#     target_path='data/',
# )

preprocess_kits19(
    source_path='/home/basia/Desktop/kits19/data',
    target_path='data/',
    labels_path='/home/basia/Desktop/kits19/data/kits.json',
)

