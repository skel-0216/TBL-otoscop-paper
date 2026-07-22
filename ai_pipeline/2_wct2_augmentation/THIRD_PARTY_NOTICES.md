# Third-party code and licenses

The style-transfer part of this pipeline reuses code from the WCT2 reference
implementation, which in turn derives part of its feature transform from
NVIDIA's FastPhotoStyle. Two licenses apply.

## WCT2 (MIT)

`model.py` (the wavelet encoder/decoder) and the bundled checkpoints
`model_checkpoints/wave_encoder_cat5_l4.pth` and `wave_decoder_cat5_l4.pth`
are from:

> Yoo, J., Uh, Y., Chun, S., Kang, B., & Ha, J. (2019).
> *Photorealistic Style Transfer via Wavelet Transforms.* ICCV.
> https://github.com/clovaai/WCT2

Copyright (c) 2019 NAVER Corp., MIT License. Full text in `LICENSE-WCT2-MIT.txt`.

`model.py` is unchanged except for one compatibility fix: the wavelet-pool
filter weights are made contiguous (`.contiguous()`) so recent PyTorch can load
the checkpoints. Values are identical.

## FastPhotoStyle (CC BY-NC-SA 4.0)

`utils/core.py` and `utils/io.py` derive from `photo_wct.py` of:

> NVIDIA FastPhotoStyle — https://github.com/NVIDIA/FastPhotoStyle
> Copyright (C) 2018 NVIDIA Corporation.

These two files are licensed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/

CC BY-NC-SA 4.0 is a **non-commercial** license with a **share-alike**
requirement. It permits academic and other non-commercial use with attribution;
it does not permit commercial use, and derivatives of these two files must be
shared under the same terms. The original attribution headers are kept at the
top of both files.

## Everything else

`pipeline.py` and the two notebooks are released with the paper. The trained
ResNet-18 weights (in `../3_classification/weights/`) are the authors' own; the
augmentation tool's licenses do not propagate to them.
