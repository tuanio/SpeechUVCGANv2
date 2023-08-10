import json
import math
import torch
import librosa
import numpy as np
import pyloudnorm as pyln
from PIL import Image
import torchaudio
import torchvision.transforms as transforms

STANDARD_LUFS = -23.0

with open("defaults.json", "r") as f:
    defaults = json.load(f)


def power_to_db(mag_spec):
    return librosa.power_to_db(mag_spec)


def db_to_power(mag_spec):
    return librosa.db_to_power(mag_spec)


def get_transform(
    opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True
):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = (opt.load_size_w, opt.load_size_h)
        transform_list.append(CustResize(osize))
    elif "scale_width" in opt.preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, opt.load_size_w, method))
        )

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __crop(img, params["crop_pos"], opt.crop_size)
                )
            )

    if opt.preprocess == "none":
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


# to convert the spectrogram ( an 2d-array of real numbers) to a storable form (0-255)
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()


# to get the original spectrogram ( an 2d-array of real numbers) from an image form (0-255)
def unscale_minmax(X, X_min, X_max, min=0.0, max=1.0):
    X = X.astype(float)
    X = (X - min) / (max - min)
    X = (X * X_max - X_min) + X_min
    return X


def extract(filename, sr=None, energy=1.0, hop_length=64, state=None):
    """
    Extracts spectrogram from an input audio file
    Arguments:
        filename: path of the audio file
        n_fft: length of the windowed signal after padding with zeros.
    """
    data, sr = librosa.load(filename, sr=sr)
    data *= energy

    ##Normalizing to standard -23.0 LuFS
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)
    data = pyln.normalize.loudness(data, loudness, target_loudness=STANDARD_LUFS)
    ##################################################

    comp_spec = librosa.stft(
        data, n_fft=defaults["n_fft"], hop_length=hop_length, window="hamming"
    )

    mag_spec, phase = librosa.magphase(comp_spec)

    phase_in_angle = np.angle(phase)
    return mag_spec, phase_in_angle, sr


def split_and_save(
    mag_spec, phase_spec, pow=1.0, state="train", channels=1, use_phase=False
):
    """
    Info: Takes a spectrogram, splits it into equal parts; uses median padding to achieve this.
    Parameters:
        mag_spec - Magnitude Spectrogram
        phase_spec - Phase Spectrogram
        pow - value to raise the magnitude spectrogram by
        state - Decides how the components are returned
        use_phase - Decides if phase spectrograms should be returned

    Modified by: Leander Maben
    """

    # because we have 129 n_fft bins; this will result in 129x128 spec components
    fix_w = defaults["fix_w"]
    orig_shape = mag_spec.shape  # mag_spec and phase_spec have same dimensions

    #### adding the padding to get equal splits
    w = orig_shape[1]
    mod_fix_w = w % fix_w
    extra_cols = 0
    if mod_fix_w != 0:
        extra_cols = fix_w - mod_fix_w

    # making padding by repeating same audio (takes care of edge case where actual data < padding columns to be added)
    num_wraps = math.ceil(extra_cols / w)
    temp_roll_mag = np.tile(mag_spec, num_wraps)
    padd_mag = temp_roll_mag[:, :extra_cols]
    mag_spec = np.concatenate((mag_spec, padd_mag), axis=1)

    temp_roll_phase = np.tile(phase_spec, num_wraps)
    padd_phase = temp_roll_phase[:, :extra_cols]
    phase_spec = np.concatenate((phase_spec, padd_phase), axis=1)
    ####

    spec_components = []

    mag_spec = power_to_db(mag_spec**pow)

    X_mag, _, _ = scale_minmax(mag_spec, 0, 255)
    X_phase, _, _ = scale_minmax(phase_spec, 0, 255)
    X_mag = np.flip(X_mag, axis=0)
    X_phase = np.flip(X_phase, axis=0)
    np_img_mag = X_mag.astype(np.uint8)
    np_img_phase = X_phase.astype(np.uint8)

    curr = [0]
    while curr[-1] < w:
        temp_spec_mag = np_img_mag[:, curr[-1] : curr[-1] + fix_w]
        temp_spec_phase = np_img_phase[:, curr[-1] : curr[-1] + fix_w]
        # rgb_im = to_rgb(temp_spec, chann = channels)
        mag_img = Image.fromarray(temp_spec_mag)
        phase_img = Image.fromarray(temp_spec_phase)
        if use_phase:
            spec_components.append([mag_img, phase_img])
        else:
            spec_components.append(mag_img)

        curr.append(curr[-1] + fix_w)

    if state == "train":
        return (
            spec_components if extra_cols == 0 else spec_components[:-1]
        )  # No need to return the component with padding.
    else:
        return spec_components  # If in "Test" state, we need all the components


def denorm_and_numpy(inp_tensor):
    # inp_tensor = inp_tensor[0, :, :, :]  # drop batch dimension
    inp_tensor = inp_tensor.permute(
        (1, 2, 0)
    )  # permute the tensor from C x H x W to H x W x C (numpy equivalent)
    inp_tensor = ((inp_tensor * 0.5) + 0.5) * 255  # to get back from transformation
    inp_tensor = inp_tensor.squeeze().numpy().astype(np.uint8)  # generating Numpy ndarray
    return inp_tensor


def processInput(filepath, power, state, channels, use_phase):
    mag_spec, phase, sr = extract(
        filepath, sr=defaults["sampling_rate"], energy=1.0, state=state
    )
    components = split_and_save(
        mag_spec, phase, pow=power, state=state, channels=channels, use_phase=use_phase
    )

    h, w = mag_spec.shape
    log_spec = power_to_db(mag_spec)
    min_mag, max_mag = log_spec.min(), log_spec.max()

    metadata = dict(
        phase=phase, size=(h, w), scale_mag=(float(min_mag), float(max_mag)), sr=sr
    )

    return components, metadata, mag_spec


def reconstruct(mag_spec, phase):
    """
    Reconstructs frames from a spectrogram and phase information.
    Arguments:
        mag_spec: Magnitude component of a spectrogram
        phase:  Phase info. of a spectrogram
    """
    temp = mag_spec * np.exp(phase * 1j)
    # data_out = torch.istft(temp, n_fft=256, hop_length=64)
    data_out = librosa.istft(temp, hop_length=64)
    # return data_out
    return data_out
