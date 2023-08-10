import argparse
import os

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.presets import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser


def parse_cmdargs():
    parser = argparse.ArgumentParser(description="Train Clean2Noisy I2I model")

    add_preset_name_parser(parser, "gen", GEN_PRESETS, "uvcgan2")
    add_preset_name_parser(parser, "head", BH_PRESETS, "bsd", "batch head")

    parser.add_argument(
        "--no-pretrain",
        dest="no_pretrain",
        action="store_true",
        help="disable uasge of the pre-trained generator",
    )

    parser.add_argument(
        "--labmda-gp",
        dest="lambda_gp",
        type=float,
        default=1.0,
        help="magnitude of the gradient penalty",
    )

    parser.add_argument(
        "--labmda-cycle",
        dest="lambda_cyc",
        type=float,
        default=5.0,
        help="magnitude of the cycle-consisntecy loss",
    )

    parser.add_argument(
        "--lr-gen",
        dest="lr_gen",
        type=float,
        default=1e-4,
        help="learning rate of the generator",
    )

    return parser.parse_args()


def get_transfer_preset(cmdargs):
    if cmdargs.no_pretrain:
        return None

    base_model = (
        "clean2noisy_output/"
        "model_m(simple-autoencoder)_d(None)"
        f"_g({GEN_PRESETS[cmdargs.gen]['model']})_pretrain-{cmdargs.gen}"
    )

    return {
        "base_model": base_model,
        "transfer_map": {
            "gen_ab": "encoder",
            "gen_ba": "encoder",
        },
        "strict": True,
        "allow_partial": False,
        "fuzzy": None,
    }


cmdargs = parse_cmdargs()
args_dict = {
    "batch_size": 64,
    "data": {
        "datasets": [
            {
                "dataset": {
                    "name": "image-domain-hierarchy",
                    "domain": domain,
                    "path": "/home/stud_vantuan/share_with_150/train_uvcganv2/CD92_5h_5h/splitted_data",
                },
                "shape": (1, 128, 128),
                "transform_train": [
                    {"name": "grayscale", "num_output_channels": 1},
                    {
                        "name": "resize",
                        "size": (128, 128),
                    },
                ],
                "transform_test": [
                    {"name": "grayscale", "num_output_channels": 1},
                    {
                        "name": "resize",
                        "size": (128, 128),
                    },
                ],
            }
            for domain in ["clean", "noisy"]
        ],
        "merge_type": "unpaired",
        "workers": 8,
    },
    "epochs": 500,
    "discriminator": {
        "model": "basic",
        "model_args": {
            "shrink_output": False,
        },
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "betas": (0.5, 0.99),
        },
        "weight_init": {
            "name": "normal",
            "init_gain": 0.02,
        },
        "spectr_norm": True,
    },
    "generator": {
        **GEN_PRESETS[cmdargs.gen],
        "optimizer": {
            "name": "Adam",
            "lr": cmdargs.lr_gen,
            "betas": (0.5, 0.99),
        },
        "weight_init": {
            "name": "normal",
            "init_gain": 0.02,
        },
    },
    "model": "uvcgan2",
    "model_args": {
        "lambda_a": cmdargs.lambda_cyc,
        "lambda_b": cmdargs.lambda_cyc,
        "lambda_idt": 0.5,
        "avg_momentum": 0.9999,
        "head_queue_size": 3,
        "head_config": {
            "name": BH_PRESETS[cmdargs.head],
            "input_features": 512,
            "output_features": 1,
            "activ": "leakyrelu",
        },
    },
    "gradient_penalty": {
        "center": 0,
        "lambda_gp": cmdargs.lambda_gp,
        "mix_type": "real-fake",
        "reduction": "mean",
    },
    "scheduler": None,
    "loss": "lsgan",
    "steps_per_epoch": 2000,
    "transfer": get_transfer_preset(cmdargs),
    # args
    "label": (
        f"{cmdargs.gen}-{cmdargs.head}_({cmdargs.no_pretrain}"
        f":{cmdargs.lambda_cyc}:{cmdargs.lambda_gp}:{cmdargs.lr_gen})"
    ),
    "outdir": os.path.join(ROOT_OUTDIR, "clean2noisy_output", "c2n"),
    "log_level": "DEBUG",
    "checkpoint": 50,
}

train(args_dict)