import os
import urllib.request
from typing import Dict, Union

from tqdm import tqdm


# TODO: replace url content(ckpt path) below.
def _pcfg(url="", hf_hub="", mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )


_RN50 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50-5d39bdab.ckpt"),
    yfcc15m=_pcfg(
        "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/open_clip/rn50_quickgelu_yfcc15m-6a5b8372.ckpt"
    ),
    cc12m=_pcfg(""),
)

_RN50_quickgelu = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50-5d39bdab.ckpt"),
    yfcc15m=_pcfg(
        "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/open_clip/rn50_quickgelu_yfcc15m-6a5b8372.ckpt"
    ),
    cc12m=_pcfg(""),
)

_RN101 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN101-a9edcaa9.ckpt"),
    yfcc15m=_pcfg(""),
)

_RN101_quickgelu = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN101-a9edcaa9.ckpt"),
    yfcc15m=_pcfg(""),
)

_RN50x4 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x4-7b8cdb29.ckpt"),
)

_RN50x16 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x16-66ea7861.ckpt"),
)

_RN50x64 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/RN50x64-839951e0.ckpt"),
)

_VITB32 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_32-34c32b89.ckpt"),
    laion400m_e31=_pcfg(""),
    laion400m_e32=_pcfg(""),
    laion2b_e16=_pcfg(""),
    laion2b_s34b_b79k=_pcfg(
        "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/open_clip/ViT_B_32_laion2b-e00a558d.ckpt"
    ),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(""),
    commonpool_m_clip_s128m_b4k=_pcfg(""),
    commonpool_m_laion_s128m_b4k=_pcfg(""),
    commonpool_m_image_s128m_b4k=_pcfg(""),
    commonpool_m_text_s128m_b4k=_pcfg(""),
    commonpool_m_basic_s128m_b4k=_pcfg(""),
    commonpool_m_s128m_b4k=_pcfg(""),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(""),
    commonpool_s_clip_s13m_b4k=_pcfg(""),
    commonpool_s_laion_s13m_b4k=_pcfg(""),
    commonpool_s_image_s13m_b4k=_pcfg(""),
    commonpool_s_text_s13m_b4k=_pcfg(""),
    commonpool_s_basic_s13m_b4k=_pcfg(""),
    commonpool_s_s13m_b4k=_pcfg(""),
)

_VITB32_quickgelu = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_32-34c32b89.ckpt"),
    laion400m_e31=_pcfg(""),
    laion400m_e32=_pcfg(""),
)

_VITB16 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_B_16-99cbeeee.ckpt"),
    laion400m_e31=_pcfg(""),
    laion400m_e32=_pcfg(""),
    laion2b_s34b_b88k=_pcfg(""),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(""),
    commonpool_l_clip_s1b_b8k=_pcfg(""),
    commonpool_l_laion_s1b_b8k=_pcfg(""),
    commonpool_l_image_s1b_b8k=_pcfg(""),
    commonpool_l_text_s1b_b8k=_pcfg(""),
    commonpool_l_basic_s1b_b8k=_pcfg(""),
    commonpool_l_s1b_b8k=_pcfg(""),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(""),
    laion400m_e32=_pcfg(""),
)

_VITL14 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_L_14-1d8bde7f.ckpt"),
    laion400m_e31=_pcfg(""),
    laion400m_e32=_pcfg(""),
    laion2b_s32b_b82k=_pcfg(""),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(""),
    commonpool_xl_clip_s13b_b90k=_pcfg(""),
    commonpool_xl_laion_s13b_b90k=_pcfg(""),
    commonpool_xl_s13b_b90k=_pcfg(""),
)

_VITL14_336 = dict(
    openai=_pcfg("https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/clip/ViT_L_14_336px-9ed46dee.ckpt"),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(
        "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/open_clip/ViT_H_14_laion2b-140b85fd.ckpt"
    ),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(""),
    laion2b_s34b_b88k=_pcfg(""),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(
        "https://download.mindspore.cn/toolkits/mindcv/mindspore-clip/open_clip/ViT_bigG_14_laion2b-d9bf227f.ckpt"
    ),
)

_robertaViTB32 = dict(
    laion2b_s12b_b32k=_pcfg(""),
)

_xlmRobertaBaseViTB32 = dict(
    laion5b_s13b_b90k=_pcfg(""),
)

_xlmRobertaLargeFrozenViTH14 = dict(
    frozen_laion5b_s13b_b90k=_pcfg(""),
)

_convnext_base = dict(
    laion400m_s13b_b51k=_pcfg(""),
)

_convnext_base_w = dict(
    laion2b_s13b_b82k=_pcfg(""),
    laion2b_s13b_b82k_augreg=_pcfg(""),
    laion_aesthetic_s13b_b82k=_pcfg(""),
)

_convnext_base_w_320 = dict(
    laion_aesthetic_s13b_b82k=_pcfg(""),
    laion_aesthetic_s13b_b82k_augreg=_pcfg(""),
)

_convnext_large_d = dict(
    laion2b_s26b_b102k_augreg=_pcfg(""),
)

_convnext_large_d_320 = dict(
    laion2b_s29b_b131k_ft=_pcfg(""),
    laion2b_s29b_b131k_ft_soup=_pcfg(""),
)

_convnext_xxlarge = dict(
    laion2b_s34b_b82k_augreg=_pcfg(""),
    laion2b_s34b_b82k_augreg_rewind=_pcfg(""),
    laion2b_s34b_b82k_augreg_soup=_pcfg(""),
)

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(""),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(""),
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(""),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(""),
)


_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "RN50x64": _RN50x64,
    "ViT-B-32": _VITB32,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-16": _VITB16,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14": _VITH14,
    "ViT-g-14": _VITg14,
    "ViT-bigG-14": _VITbigG14,
    "roberta-ViT-B-32": _robertaViTB32,
    "xlm-roberta-base-ViT-B-32": _xlmRobertaBaseViTB32,
    "xlm-roberta-large-ViT-H-14": _xlmRobertaLargeFrozenViTH14,
    "convnext_base": _convnext_base,
    "convnext_base_w": _convnext_base_w,
    "convnext_base_w_320": _convnext_base_w_320,
    "convnext_large_d": _convnext_large_d,
    "convnext_large_d_320": _convnext_large_d_320,
    "convnext_xxlarge": _convnext_xxlarge,
    "coca_ViT-B-32": _coca_VITB32,
    "coca_ViT-L-14": _coca_VITL14,
    "EVA01-g-14": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
        laion400m_s11b_b41k=_pcfg(""),
    ),
    "EVA01-g-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt
        merged2b_s11b_b114k=_pcfg(""),
    ),
    "EVA02-B-16": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt
        merged2b_s8b_b131k=_pcfg(""),
    ),
    "EVA02-L-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt
        merged2b_s4b_b131k=_pcfg(""),
    ),
    "EVA02-L-14-336": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
        merged2b_s6b_b61k=_pcfg(""),
    ),
    "EVA02-E-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt
        laion2b_s4b_b115k=_pcfg(""),
    ),
    "EVA02-E-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt
        laion2b_s9b_b144k=_pcfg(""),
    ),
}


def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace("-", "_")


def list_pretrained(as_str: bool = False):
    """returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [":".join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag: str):
    """return all models having the specified pretrain tag"""
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_tags_by_model(model: str):
    """return all pretrain tags for the specified model architecture"""
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return _clean_tag(tag) in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str):
    cfg = get_pretrained_cfg(model, _clean_tag(tag))
    return cfg.get("url", "")


def download_pretrained_from_url(
    url: str,
    cache_dir: Union[str, None] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit="iB", unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def download_pretrained(
    cfg: Dict,
    cache_dir: Union[str, None] = None,
):
    target = ""
    if not cfg:
        return target

    download_url = cfg.get("url", "")
    if download_url:
        target = download_pretrained_from_url(download_url, cache_dir=cache_dir)

    return target
