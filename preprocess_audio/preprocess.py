import argparse
import utils
import os
import numpy as np
import glob
import torch
import concurrent.futures as cf
from PIL import Image
import librosa
import math
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, vflip

# TTF.vflip

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src-path",
    type=str,
    default=r"F:\astar\works\augment asr model\ganspeechaugment\test\test_audio",
)  # audio path
parser.add_argument(
    "--tgt-magnitude-path",
    type=str,
    default=r"F:\astar\works\augment asr model\ganspeechaugment\test\save_fold\magnitude",
)  # where to save spectrogram image
parser.add_argument(
    "--tgt-metadata-path",
    type=str,
    default=r"F:\astar\works\augment asr model\ganspeechaugment\test\save_fold\metadata",
)  # where to save spectrogram image
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--state", type=str, default="train")  # train or test
args = parser.parse_args()

wavs = glob.glob(os.path.join(args.src_path, "*.wav"))

FIX_W = 128
N_FFT = 255


# to convert the spectrogram ( an 2d-array of real numbers) to a storable form (0-255)
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()


def extract(file):
    data, sr = librosa.load(file, sr=None)
    comp_spec = librosa.stft(data, n_fft=N_FFT, hop_length=64, window="hamming")
    mag_spec, phase = librosa.magphase(comp_spec)

    phase_in_angle = np.angle(phase)

    w = mag_spec.shape[1]
    mod_fix_w = w % FIX_W
    extra_cols = 0
    if mod_fix_w != 0:
        extra_cols = FIX_W - mod_fix_w

    num_wraps = math.ceil(extra_cols / w)
    temp_roll_mag = np.tile(mag_spec, num_wraps)
    padd_mag = temp_roll_mag[:, :extra_cols]
    mag_spec = np.concatenate((mag_spec, padd_mag), axis=1)
    mag_scaled, _min, _max = scale_minmax(mag_spec, 0, 255)

    spec_components = []

    curr = [0]
    while curr[-1] < w:
        temp_spec_mag = mag_scaled[:, curr[-1] : curr[-1] + FIX_W]
        mag_spec_comp = Image.fromarray(temp_spec_mag.astype(np.uint8))
        spec_components.append(mag_spec_comp)
        curr.append(curr[-1] + FIX_W)

    h, w = phase.shape

    metadata = dict(phase=phase, size=(h, w), sr=sr, scale_param=(_min, _max))

    return spec_components, metadata


def componentize(file):
    name = file.rsplit(os.sep, 1)[-1].rsplit(".", 1)[0]
    mag_components, metadata = extract(file)
    for idx, mag in enumerate(mag_components):
        mag.save(os.path.join(args.tgt_magnitude_path, f"{name}_{idx}_mag.png"), "PNG")
    torch.save(metadata, os.path.join(args.tgt_metadata_path, f"{name}_metadata.pt"))


with cf.ThreadPoolExecutor(max_workers=args.threads) as exe:
    list(exe.map(componentize, wavs))
