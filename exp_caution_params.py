from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint


from MBExperiment import MBExperiment
from MPC import MPC
from config import create_config

import env  # this is run so that the env is registered

import torch
import numpy as np

# use multiprocessing to speed up for-loops
from multiprocessing import Pool


class create_args:
    def __init__(
        self,
        CARL,
        caution_param,
        test_domain,
        pretrain_dir=None,
        record_video=False,
    ):
        """
        creates class to pass on correct arguments to MPExperiment.py

        arguments:
            CARL: str
                CARL State or CARL Reward as adaptation algorithm
            caution_param: float
                set caution parameter gamma for CARL Reward/ lambda_2 for CARL State
            test_domain: float
                value for the test domain for the environment (pole length)
            pretrain_dir: str
                directory from which model weights will be loaded if pretraining is completed
            record_video: Boolean
                whether to record the test rollouts
        """

        self.test_domain = test_domain
        self.record_video = record_video

        # scaler for penalty during adaptation (for catastrophic state risk-aversion; lambda_2)
        self.penalty_scale = 1
        # during pretraining, use PETS (= CARL Reward with gamma=100)
        self.percentile = 100
        # number of training iterations, defaults to the one defined in config/cartpole.py
        self.ntrain_iters = None
        # number of networks in the ensemble
        self.num_nets = 5
        # either complete full pretraining or start with adaptation
        self.start_epoch = 0
        self.continue_train = False
        self.test_mode = False

        # set training/test parameters
        # number of initial random rollouts
        self.ninit_rollouts = 1
        # number of rollouts per training iteration
        self.nitr_per_rollout = 0
        # number of test rollouts
        self.ntest_rollouts = 1
        # number of adaptation iterations to perform on test environment
        self.nadapt_iters = 10

        # directory to which model/videos are loaded
        self.logdir = "log"
        # use MPC
        self.ctrl_type = "MPC"
        # use cartpole environments
        self.env = "cartpole"
        # use CEMOptimizer
        self.optimizer = "CEM"

        # create suffix to attach to a run
        caution_param_str = str(caution_param).replace(".", "_")
        td_str = str(test_domain).replace(".", "_")
        self.suffix = f"{CARL}_{caution_param_str}_td_{td_str}"

        if CARL == "State":
            # enable catastrophe prediction state safety labels
            self.no_catastrophe_pred = False
            self.penalty_scale = caution_param
            # test percentile acts as beta for CARL State
            self.test_percentile = 50
        elif CARL == "Reward":
            # disable catastrophe prediction and training for state safety labels
            self.no_catastrophe_pred = True
            # test percentile acts as gamma for CARL Reward
            # unlike in original code, here, gamma is *subtracted* from 100 for closer correspondence to the paper
            self.test_percentile = 100 - caution_param

        # if pretraining directory is given, assume that only adaptation is performed
        if pretrain_dir is not None:
            self.test_mode = True
            self.ninit_rollouts = 0
            self.ntrain_iters = 0
            self.continue_train = True
        self.load_model_dir = pretrain_dir


def run_CARL(CARL, caution_param, test_domain, pretrain_dir, record_video):
    """
    runs (pretraining and) adaptation with CARL.

    arguments:
        CARL: str
            CARL State or CARL Reward as adaptation algorithm
        caution_param: float
            set caution parameter gamma for CARL Reward/ lambda_2 for CARL State
        test_domain: float
            value for the test domain for the environment (pole length)
        pretrain_dir: str
            directory from which model weights will be loaded if pretraining is completed
        record_video: Boolean
            whether to record the test rollouts
    """
    args = create_args(
        CARL,
        caution_param,
        test_domain,
        pretrain_dir=pretrain_dir,
        record_video=record_video,
    )

    cfg = create_config(args)
    cfg.pprint()

    assert args.ctrl_type == "MPC"

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    if args.load_model_dir is not None:
        exp.policy.model.load_state_dict(
            torch.load(os.path.join(args.load_model_dir, "weights"))
        )
    print(args.load_model_dir)
    if not os.path.exists(exp.logdir):
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    train_dir = exp.run_experiment()

    return train_dir


def main(caution_param_args):
    """
    use parsed arguments to loop over caution parameter settings and run CARL

    arguments:
    caution_param_args:
        parsed arguments from command line
    """

    # loop over td
    range_caution_params = np.linspace(
        caution_param_args.min_caution,
        caution_param_args.max_caution,
        caution_param_args.ncaution_params,
    )
    range_test_domains = np.linspace(
        caution_param_args.min_td,
        caution_param_args.max_td,
        caution_param_args.ntds,
    )

    # call the same function with different data in parallel
    with Pool() as pool:
        if caution_param_args.pretrain_dir is None:
            # pretrain model and use first adaptation setting
            pretrain_dir = run_CARL(
                caution_param_args.CARL,
                range_caution_params[0],
                range_test_domains[0],
                None,
                caution_param_args.record_video,
            )
            # prepare items
            items_pool = [
                (
                    caution_param_args.CARL,
                    caution_param,
                    test_domain,
                    pretrain_dir,
                    caution_param_args.record_video,
                )
                for caution_param in range_caution_params
                for test_domain in range_test_domains
            ]
            # use all except the first adaptation settings in parallel on same pretraining
            pool.starmap_async(run_CARL, items_pool[1:])

        else:
            pretrain_dir = caution_param_args.pretrain_dir
            # prepare arguments
            items_pool = [
                (
                    caution_param_args.CARL,
                    caution_param,
                    test_domain,
                    pretrain_dir,
                    caution_param_args.record_video,
                )
                for caution_param in range_caution_params
                for test_domain in range_test_domains
            ]
            # use all adaptation settings in parallel on same pretraining
            pool.starmap(run_CARL, items_pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--CARL",
        type=str,
        required=True,
        help="version of CARL for adaptation: select from [State, Reward], note: it is recommended to train CARL State first since the same pretraining can be used for CARL Reward but not vice versa.",
    )
    parser.add_argument(
        "--min_caution",
        type=float,
        required=True,
        help="minimum value for caution parameter gamma for CARL Reward or lambda_2 for CARL State",
    )
    parser.add_argument(
        "--max_caution",
        type=float,
        required=True,
        help="max value for caution parameter gamma for CARL Reward or lambda_2 for CARL State",
    )
    parser.add_argument(
        "--ncaution_params",
        type=int,
        required=True,
        help="number of caution parameter settings considered in [min_caution, max_caution]",
    )
    parser.add_argument(
        "--min_td",
        default=1,
        type=float,
        help="minimum value for considered test domains",
    )
    parser.add_argument(
        "--max_td",
        default=2,
        type=float,
        help="max value for considered test domains",
    )
    parser.add_argument(
        "--ntds",
        type=int,
        default=3,
        help="number of test domains considered in [min_td, max_td]",
    )
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default=None,
        help="if pretraining was already conducted, provide path to weights",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="whether to record the test rollouts",
    )

    caution_param_args = parser.parse_args()

    main(caution_param_args)
