import argparse
import utils
import torch
import torchaudio
from utils import denorm_and_numpy
import soundfile as sf
import os
import numpy as np
import glob
import concurrent.futures as cf
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, vflip

# TTF.vflip

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tgt-audio-path",
    type=str,
    default=r"F:\astar\works\augment asr model\ganspeechaugment\test\save_fold\results",
)  # audio path

# magnitude here is the one translated
parser.add_argument(
    "--src-magnitude-path",
    type=str,
    default=r"F:\astar\works\augment asr model\ganspeechaugment\test\save_fold\magnitude",
)  # where to save spectrogram image

# phase here is original phase
parser.add_argument(
    "--src-metadata-path",
    type=str,
    default=r"F:\astar\works\augment asr model\ganspeechaugment\test\save_fold\metadata",
)  # where to save spectrogram image
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--state", type=str, default="train")  # train or test
parser.add_argument("--fix-w", type=int, default=128)
parser.add_argument("--n-fft", type=int, default=255)
parser.add_argument("--energy", type=float, default=1.0)
parser.add_argument("--power", type=float, default=1.0)
args = parser.parse_args()

FIX_W = args.fix_w
N_FFT = args.n_fft
metadata_path = args.src_metadata_path

# to get the original spectrogram ( an 2d-array of real numbers) from an image form (0-255)
def unscale_minmax(X, X_min, X_max, min=0.0, max=1.0):
    X = X.astype(float)
    X = (X - min) / (max - min)
    X = (X * X_max - X_min) + X_min
    return X

list_mag = glob.glob(os.path.join(args.src_magnitude_path, "*_mag.png"))
list_mag = sorted(list_mag)

data_dict = defaultdict(list)
for mag_file in tqdm(list_mag, desc="Loading magnitude"):
    name = mag_file.rsplit(os.sep, 1)[-1].split("_mag.png")[0]
    root_name = name.rsplit("_", 1)[0]
    mag = np.asarray(Image.open(mag_file)).astype(float)
    # k = pil_to_tensor(Image.open(mag_file))
    # mag = denorm_and_numpy(k)
    # mag = Image.fromarray(mag).resize((FIX_W, N_FFT // 2 + 1), Image.Resampling.LANCZOS)
    # mag = np.array(mag).astype(float)
    data_dict[root_name].append(mag)

audio_list = []
for k, v in tqdm(data_dict.items(), "Concatenating..."):
    audio_list.append({"name": k, "mag": np.concatenate(v, axis=1)})

def combine(pack):
    metadata = torch.load(os.path.join(metadata_path, pack["name"] + "_metadata.pt"))
    h, w = metadata["size"]
    im_mag = pack['mag']
    _min, _max = metadata['scale_param']
    mod_fix_w = w % FIX_W
    if mod_fix_w != 0:
        im_mag = im_mag[:, : -(FIX_W - mod_fix_w)]
    im_mag = unscale_minmax(im_mag, _min, _max, 0, 255)
    audio = utils.reconstruct(im_mag, metadata["phase"]) / args.energy
    sf.write(
        os.path.join(args.tgt_audio_path, pack["name"] + ".wav"),
        audio,
        metadata["sr"],
    )


# def combine(pack):
#     metadata = torch.load(os.path.join(metadata_path, pack["name"] + "_metadata.pt"))
#     h, w = metadata["size"]
#     min_mag, max_mag = metadata["scale_mag"]
#     im_mag = pack['mag']
#     mod_fix_w = w % FIX_W
#     if mod_fix_w != 0:
#         im_mag = im_mag[:, : -(FIX_W - mod_fix_w)]
#     im_mag = np.flip(im_mag, axis=0)
#     im_mag = utils.unscale_minmax(im_mag, min_mag, max_mag, 0, 255)
#     im_mag = utils.db_to_power(im_mag)
#     # im_mag = np.power(im_mag, 1.0 / args.power)
#     Image.fromarray(utils.power_to_db(im_mag).astype(np.uint8)).save(r'F:\astar\works\augment asr model\ganspeechaugment\test\save_fold\combined_mag_spec' + os.sep + pack['name'] + '.png', 'PNG')
#     audio = utils.reconstruct(im_mag, metadata["phase"]) / args.energy
#     sf.write(
#         os.path.join(args.tgt_audio_path, pack["name"] + ".wav"),
#         audio,
#         metadata["sr"],
#     )


with cf.ThreadPoolExecutor(max_workers=args.threads) as exe:
    list(exe.map(combine, audio_list))
