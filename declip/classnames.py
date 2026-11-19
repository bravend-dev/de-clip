"""Class names of the ten evaluation datasets (Table 2 of the paper).

Names are used verbatim inside the label templates (Sec. 3.3) and as keys into the
description files under ``descriptions/`` (Sec. 3.4). Order matches the integer
labels in the evaluation CSV files, so it must not be changed.
"""

from __future__ import annotations

PAD: list[str] = [
    "nevus",
    "basal cell carcinoma",
    "actinic keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "melanoma",
]

HAM: list[str] = [
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "vascular lesion",
]

F17K: list[str] = [
    "dermatofibroma",
    "psoriasis",
    "kaposi sarcoma",
    "neutrophilic dermatoses",
    "granuloma annulare",
    "naevus comedonicus",
    "milia",
    "telangiectases",
    "lichen planus",
    "perioral dermatitis",
    "paronychia",
    "hidradenitis",
    "neurodermatitis",
    "necrobiosis lipoidica",
    "superficial spreading melanoma ssm",
    "actinic keratosis",
    "basal cell carcinoma",
    "acne",
    "acne vulgaris",
    "allergic contact dermatitis",
    "acquired autoimmune bullous diseaseherpes gestationis",
    "rhinophyma",
    "porokeratosis actinic",
    "lupus subacute",
    "livedo reticularis",
    "folliculitis",
    "disseminated actinic porokeratosis",
    "pityriasis rubra pilaris",
    "porphyria",
    "rosacea",
    "melanoma",
    "sarcoidosis",
    "squamous cell carcinoma",
    "prurigo nodularis",
    "tuberous sclerosis",
    "lentigo maligna",
    "erythema annulare centrifigum",
    "urticaria",
    "pustular psoriasis",
    "congenital nevus",
    "dariers disease",
    "dyshidrotic eczema",
    "eczema",
    "mycosis fungoides",
    "sun damaged skin",
    "urticaria pigmentosa",
    "malignant melanoma",
    "fixed eruptions",
    "pyogenic granuloma",
    "stevens johnson syndrome",
    "scabies",
    "neurofibromatosis",
    "erythema multiforme",
    "aplasia cutis",
    "behcets disease",
    "nevocytic nevus",
    "langerhans cell histiocytosis",
    "vitiligo",
    "dermatomyositis",
    "pityriasis rosea",
    "fordyce spots",
    "scleromyxedema",
    "pityriasis lichenoides chronica",
    "tick bite",
    "scleroderma",
    "nevus sebaceous of jadassohn",
    "seborrheic keratosis",
    "stasis edema",
    "lyme disease",
    "porokeratosis of mibelli",
    "xanthomas",
    "pediculosis lids",
    "becker nevus",
    "striae",
    "juvenile xanthogranuloma",
    "lupus erythematosus",
    "acrodermatitis enteropathica",
    "nematode infection",
    "basal cell carcinoma morpheiform",
    "photodermatoses",
    "incontinentia pigmenti",
    "drug induced pigmentary changes",
    "granuloma pyogenic",
    "port wine stain",
    "keratosis pilaris",
    "mucinosis",
    "papilomatosis confluentes and reticulate",
    "syringoma",
    "acanthosis nigricans",
    "drug eruption",
    "xeroderma pigmentosum",
    "halo nevus",
    "lichen amyloidosis",
    "ehlers danlos syndrome",
    "solid cystic basal cell carcinoma",
    "epidermal nevus",
    "ichthyosis vulgaris",
    "keloid",
    "mucous cyst",
    "calcinosis cutis",
    "lymphangioma",
    "seborrheic dermatitis",
    "pilomatricoma",
    "pilar cyst",
    "erythema nodosum",
    "neurotic excoriations",
    "hailey hailey disease",
    "erythema elevatum diutinum",
    "factitial dermatitis",
    "tungiasis",
    "epidermolysis bullosa",
    "myiasis",
    "cheilitis",
    "lichen simplex",
]

DAFFODIL: list[str] = [
    "vitiligo",
    "acne",
    "SJS/TEN",
    "hyperpigmentation",
    "nail psoriasis",
]

BCN20000: list[str] = [
    "melanoma",
    "squamous cell carcinoma",
    "nevus",
    "basal cell carcinoma",
    "seborrheic keratosis",
    "actinic keratosis",
    "solar lentigo",
    "dermatofibroma",
    "vascular lesion",
]

DDI: list[str] = [
    "benign",
    "malignant",
]

HIBA: list[str] = [
    "nevus",
    "basal cell carcinoma",
    "vascular lesion",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "actinic keratosis",
    "dermatofibroma",
    "solar lentigo",
    "lichenoid keratosis",
    "melanoma",
]

MSKCC: list[str] = [
    "benign",
    "malignant",
]

DERMNET: list[str] = [
    "acne and rosacea photos",
    "actinic keratosis basal cell carcinoma and other malignant lesions",
    "atopic dermatitis photos",
    "bullous disease photos",
    "cellulitis impetigo and other bacterial infections",
    "eczema photos",
    "exanthems and drug eruptions",
    "hair loss photos alopecia and other hair diseases",
    "herpes hpv and other stds photos",
    "light diseases and disorders of pigmentation",
    "lupus and other connective tissue diseases",
    "melanoma skin cancer nevi and moles",
    "nail fungus and other nail disease",
    "poison ivy photos and other contact dermatitis",
    "psoriasis pictures lichen planus and related diseases",
    "scabies lyme disease and other infestations and bites",
    "seborrheic keratoses and other benign tumors",
    "systemic disease",
    "tinea ringworm candidiasis and other fungal infections",
    "urticaria hives",
    "vascular tumors",
    "vasculitis photos",
    "warts molluscum and other viral infections",
]

DERM7PT: list[str] = [
    "benign",
    "malignant",
]

CLASSNAMES_BY_DATASET: dict[str, list[str]] = {
    "PAD": PAD,
    "HAM": HAM,
    "F17K": F17K,
    "Daffodil": DAFFODIL,
    "BCN20000": BCN20000,
    "DDI": DDI,
    "HIBA": HIBA,
    "MSKCC": MSKCC,
    "Dermnet": DERMNET,
    "Derm7pt": DERM7PT,
}


def all_classnames() -> list[str]:
    """Sorted union of the class names over all datasets (147 names)."""
    names: set[str] = set()
    for values in CLASSNAMES_BY_DATASET.values():
        names.update(values)
    return sorted(names)
