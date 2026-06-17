"""Tests for the hierarchical, multi-axis ontology layer (Theme L: Tasks 36-39).

These cover the additive ontology features without asserting anything about emitted JSONL: the
RadLex hierarchy queries, the multi-axis decomposition, the secondary-ontology cross-codes, and the
registry-validation audit. They also pin the backwards-compatibility guarantee that no existing id
or RadLex name changed.
"""

from config import labels, masks
from config.ontology import (
    OntologyTerm,
    ancestors,
    descendants,
    descendants_of,
    group_by_level,
    roots,
)

# RadLex ids of concepts that legitimately have no RadLex code (documented in the config files);
# the validation audit is allowed to report exactly these as "empty id" and nothing more.
KNOWN_EMPTY_ID_LABELS = {"TransitionalCellCarcinoma"}
KNOWN_EMPTY_ID_MASKS = {"Background"}
# Pre-existing duplicate RadLex ids in the source data; surfaced (not silently fixed) per Task 38.
KNOWN_DUPLICATE_IDS = {"RID5352", "RID5231"}


def test_kidney_tumour_subtree_ancestors():
    """Clear-cell adenocarcinoma resolves up the full RadLex chain to Neoplasm (Task 36)."""
    cc = labels.label_by_name("ClearCellAdenocarcinoma")
    chain = [a.radlex_name for a in labels.label_ancestors(cc)]
    assert chain == ["RenalAdenocarcinoma", "Adenocarcinoma", "Malignant", "Neoplasm"]


def test_neoplasm_descendants_cover_every_subtype():
    """Every kidney-tumour subtype is reachable below Neoplasm (Task 36)."""
    names = {d.radlex_name for d in labels.label_descendants_of("Neoplasm")}
    for expected in [
        "Malignant",
        "Benign",
        "Adenocarcinoma",
        "RenalAdenocarcinoma",
        "ClearCellAdenocarcinoma",
        "ChromophobeAdenocarcinoma",
        "PapillaryRenalAdenocarcinoma",
        "MultilocularCysticRenalTumor",
        "WilmsTumor",
        "Angiomyolipoma",
        "Oncocytoma",
        "RenalCyst",
    ]:
        assert expected in names


def test_chest_pathology_subtree():
    """The chest subtree links viral pneumonia under pneumonia (Task 36)."""
    pv = labels.label_by_name("PneumoniaViral")
    assert [a.radlex_name for a in labels.label_ancestors(pv)] == ["Pneumonia"]


def test_group_by_level_roots_have_no_parent():
    """Level 0 of the hierarchy is exactly the parentless labels."""
    levels = labels.labels_grouped_by_level()
    assert 0 in levels
    root_names = {label.radlex_name for label in roots(labels.all_labels)}
    assert {label.radlex_name for label in levels[0]} == root_names


def test_multi_axis_decomposition():
    """A labelled finding decomposes into anatomy / pathology / modifier axes (Task 37)."""
    cc = labels.label_by_name("ClearCellAdenocarcinoma")
    assert cc.anatomy.radlex_name == "Kidney"
    assert cc.pathology.radlex_name == "Neoplasm"
    assert cc.modifier.radlex_name == "Malignant"
    # source name carried through on the pathology axis
    assert cc.pathology.source_name == "clear_cell_rcc"


def test_axes_default_to_none_and_are_optional():
    """Labels without an explicit decomposition keep ``None`` axes (additive, opt-in)."""
    normal = labels.label_by_name("NormalityDecriptor")
    assert normal.anatomy is None and normal.pathology is None and normal.modifier is None


def test_secondary_ontology_codes(in_subset=("Kidney", "Lung", "Liver", "Brain")):
    """Anatomy masks carry optional SNOMED/FMA/Uberon cross-codes (Task 39)."""
    kidney = masks.mask_by_name("Kidney")
    assert kidney.snomed_id == "64033007"
    assert kidney.fma_id == "FMA:7203"
    assert kidney.uberon_id == "UBERON:0002113"
    # every documented organ mask is populated
    for name in in_subset:
        assert masks.mask_by_name(name).uberon_id


def test_organ_masks_query():
    """The organ-mask query returns the anatomy-axis masks (Task 36 'all organ masks')."""
    organ_names = {m.radlex_name for m in masks.organ_masks()}
    assert {"Kidney", "Lung", "Liver", "Brain"} <= organ_names


def test_validate_reports_only_known_issues():
    """The audit surfaces (rather than hides) the known data issues, and introduces no new ones."""
    label_issues = labels.validate_labels()
    # Every reported empty-id label is a known, documented one.
    empty_id_labels = {msg.split("'")[1] for msg in label_issues if msg.startswith("Empty RadLex id")}
    assert empty_id_labels == KNOWN_EMPTY_ID_LABELS
    # Every duplicate-id report is a known pre-existing duplicate.
    for msg in label_issues:
        if msg.startswith("Duplicate RadLex id"):
            assert any(dup in msg for dup in KNOWN_DUPLICATE_IDS)
    # No dangling parents or cycles introduced by the hierarchy wiring.
    assert not [m for m in label_issues if "Dangling parent" in m or "cycle" in m]

    mask_issues = masks.validate_masks()
    empty_id_masks = {msg.split("'")[1] for msg in mask_issues if msg.startswith("Empty RadLex id")}
    assert empty_id_masks == KNOWN_EMPTY_ID_MASKS
    assert not [m for m in mask_issues if "Dangling parent" in m or "cycle" in m]


def test_generic_helpers_work_on_both_registries():
    """The generic ontology helpers operate uniformly over labels and masks."""
    assert {d.radlex_name for d in descendants_of("Neoplasm", masks.all_masks)} >= {"Metastasis"}
    neoplasm_label = labels.label_by_name("Neoplasm")
    assert neoplasm_label in roots(labels.all_labels)
    assert group_by_level(masks.all_masks)  # non-empty


def test_backwards_compatible_ids_and_names_unchanged():
    """Adding the ontology layer must not change any existing id, name, or registry size."""
    assert len(labels.all_labels) == 48
    assert len(masks.all_masks) == 17
    # spot-check a few canonical (id, radlex_name) pairs that downstream output depends on
    assert (labels.Neoplasm.id, labels.Neoplasm.radlex_name) == (1, "Neoplasm")
    assert (masks.Kidney.id, masks.Kidney.color, masks.Kidney.radlex_name) == (1, 1, "Kidney")


def test_ontology_term_is_empty():
    """An all-blank OntologyTerm reports empty; any populated field flips it."""
    assert OntologyTerm().is_empty()
    assert not OntologyTerm(radlex_name="Kidney").is_empty()


def test_ancestors_descendants_consistency():
    """If B is a descendant of A then A is an ancestor of B (relationship is symmetric)."""
    neoplasm = labels.label_by_name("Neoplasm")
    for child in descendants(neoplasm, labels.all_labels):
        assert neoplasm in ancestors(child, labels.all_labels)
