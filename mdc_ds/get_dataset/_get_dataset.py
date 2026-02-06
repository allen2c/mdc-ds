from typing import TYPE_CHECKING, Literal, Tuple, TypeAlias

if TYPE_CHECKING:
    from datasets import Dataset


def get_dataset(
    name: "DatasetNameType",
    split: Literal["train", "test", "validation"] = "train",
) -> "Dataset":
    if name == "common-voice-v24-english-en-au-subset-fo-0447e8a6":
        from .common_voice_v24_english_en_au_subset_fo_0447e8a6 import (
            get_dataset as get_dataset_en_au,
        )

        return get_dataset_en_au(name=name, split=split)

    elif name == "mcv-scripted-de-v24.0":
        from .mcv_scripted_de_v24_0 import get_dataset as get_dataset_de

        return get_dataset_de(name=name, split=split)

    elif name == "mcv-scripted-es-v24.0":
        from .mcv_scripted_es_v24_0 import get_dataset as get_dataset_es

        return get_dataset_es(name=name, split=split)

    elif name == "mcv-scripted-fr-v24.0":
        from .mcv_scripted_fr_v24_0 import get_dataset as get_dataset_fr

        return get_dataset_fr(name=name, split=split)

    elif name == "mcv-scripted-id-v24.0":
        from .mcv_scripted_id_v24_0 import get_dataset as get_dataset_id

        return get_dataset_id(name=name, split=split)

    elif name == "mcv-scripted-it-v24.0":
        from .mcv_scripted_it_v24_0 import get_dataset as get_dataset_it

        return get_dataset_it(name=name, split=split)

    elif name == "mcv-scripted-ja-v24.0":
        from .mcv_scripted_ja_v24_0 import get_dataset as get_dataset_ja

        return get_dataset_ja(name=name, split=split)

    elif name == "mcv-scripted-ko-v24.0":
        from .mcv_scripted_ko_v24_0 import get_dataset as get_dataset_ko

        return get_dataset_ko(name=name, split=split)

    elif name == "mcv-scripted-ms-v24.0":
        from .mcv_scripted_ms_v24_0 import get_dataset as get_dataset_ms

        return get_dataset_ms(name=name, split=split)

    elif name == "mcv-scripted-nan-tw-v24.0":
        from .mcv_scripted_nan_tw_v24_0 import get_dataset as get_dataset_nan_tw

        return get_dataset_nan_tw(name=name, split=split)

    elif name == "mcv-scripted-ru-v24.0":
        from .mcv_scripted_ru_v24_0 import get_dataset as get_dataset_ru

        return get_dataset_ru(name=name, split=split)

    elif name == "mcv-scripted-th-v24.0":
        from .mcv_scripted_th_v24_0 import get_dataset as get_dataset_th

        return get_dataset_th(name=name, split=split)

    elif name == "mcv-scripted-vi-v24.0":
        from .mcv_scripted_vi_v24_0 import get_dataset as get_dataset_vi

        return get_dataset_vi(name=name, split=split)

    elif name == "mcv-scripted-zh-TW-v24.0":
        from .mcv_scripted_zh_tw_v24_0 import get_dataset as get_dataset_zh_tw

        return get_dataset_zh_tw(name=name, split=split)

    elif name == "mcv-scripted-zh-CN-v24.0":
        from .mcv_scripted_zh_cn_v24_0 import get_dataset as get_dataset_zh_cn

        return get_dataset_zh_cn(name=name, split=split)

    elif name == "mcv-scripted-zh-HK-v24.0":
        from .mcv_scripted_zh_hk_v24_0 import get_dataset as get_dataset_zh_hk

        return get_dataset_zh_hk(name=name, split=split)

    else:
        raise ValueError(f"Invalid dataset name: {name}")


DatasetNameType: TypeAlias = Literal[
    "common-voice-v24-english-en-au-subset-fo-0447e8a6",
    "mcv-scripted-de-v24.0",
    "mcv-scripted-es-v24.0",
    "mcv-scripted-fr-v24.0",
    "mcv-scripted-id-v24.0",
    "mcv-scripted-it-v24.0",
    "mcv-scripted-ja-v24.0",
    "mcv-scripted-ko-v24.0",
    "mcv-scripted-ms-v24.0",
    "mcv-scripted-nan-tw-v24.0",
    "mcv-scripted-ru-v24.0",
    "mcv-scripted-th-v24.0",
    "mcv-scripted-vi-v24.0",
    "mcv-scripted-zh-CN-v24.0",
    "mcv-scripted-zh-HK-v24.0",
    "mcv-scripted-zh-TW-v24.0",
]
implemented_dataset_names: Tuple[DatasetNameType, ...] = (
    "common-voice-v24-english-en-au-subset-fo-0447e8a6",
    "mcv-scripted-de-v24.0",
    "mcv-scripted-es-v24.0",
    "mcv-scripted-fr-v24.0",
    "mcv-scripted-id-v24.0",
    "mcv-scripted-it-v24.0",
    "mcv-scripted-ja-v24.0",
    "mcv-scripted-ko-v24.0",
    "mcv-scripted-ms-v24.0",
    "mcv-scripted-nan-tw-v24.0",
    "mcv-scripted-ru-v24.0",
    "mcv-scripted-th-v24.0",
    "mcv-scripted-vi-v24.0",
    "mcv-scripted-zh-CN-v24.0",
    "mcv-scripted-zh-HK-v24.0",
    "mcv-scripted-zh-TW-v24.0",
)
