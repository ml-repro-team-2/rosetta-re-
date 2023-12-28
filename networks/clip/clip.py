import hashlib
import os
import urllib
import warnings
from typing import Union
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms

from .model import build_model

__all__ = ["available_models", "load", "clip_mean", "clip_std"]

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}

clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _transform(img):
    """
    PIL image to torch.Tensor with clip's normalization params
    for inputting a PIL image into the clip model
    """
    return transforms.Compose(
        [
            transforms.Resize(img, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                clip_mean,
                clip_std,
            ),
        ]
    )


def available_models():
    # links of these models are available.. for other models we need to download their pytorch checkpoints (like from timm)
    return list(_MODELS.keys())


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=False,
):
    """
    Load one of the CLIP models (resnet50, resnet101, resnet50x4, vitb32) or any custom model from its checkpoint.

    Parameters
    ----------
    name : str
        either name of the model or path of its checkpoint (if it is a custom model)

    Returns
    -------
    model : nn.Module
        the clip model

    preprocess :
        transforms PIL image into tensor supported by CLIP models
    """

    if jit:
        raise RuntimeError()

    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found.\nCheck the available models :\n{available_models()}"
        )

    # load the variable
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except:
        warnings.warn("Model is not a jit archive")
        state_dict = torch.load(model_path, map_location="cpu").eval()
        
    model = build_model(state_dict or model.state_dict()).to(device)

    if str(device) == "cpu":
        model.float()

    return model, _transform(model.visual.input_resolution)
