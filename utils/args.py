import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch PPO example")
    parser.add_argument(
        "--env-name",
        default="Hopper-v2",
        metavar="G",
        help="name of the environment to run",
    )
    parser.add_argument(
        "--model-path", 
        metavar="G", 
        help="path of pre-trained model"
        )
    parser.add_argument(
        "--render", action="store_true", default=False, help="render the environment"
    )
    parser.add_argument(
        "--log-std",
        type=float,
        default=-0.0,
        metavar="G",
        help="log std for the policy (default: -0.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.95, metavar="G", help="gae (default: 0.95)"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=1e-3,
        metavar="G",
        help="l2 regularization regression (default: 1e-3)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        metavar="G",
        help="learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--randomization", 
        type=int, 
        default=1, 
        metavar="G"
    )
    parser.add_argument(
        "--adaptation", 
        type=int, 
        default=1, 
        metavar="G"
    )
    parser.add_argument(
        "--depth-prediction-model", 
        default="dpt", 
        metavar="G"
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        metavar="N",
        help="clipping epsilon for PPO",
    )
    parser.add_argument(
        "--hist-length",
        type=int,
        default=4,
        metavar="N",
        help="the number of consecutive history infos (default: 4)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        metavar="N",
        help="number of threads for agent (default: 4)",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        metavar="N", 
        help="random seed (default: 1)"
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=2048,
        metavar="N",
        help="minimal batch size per PPO update (default: 2048)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2048,
        metavar="N",
        help="minimal batch size for evaluation (default: 2048)",
    )
    parser.add_argument(
        "--max-iter-num",
        type=int,
        default=200,
        metavar="N",
        help="maximal number of main iterations (default: 500)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )
    parser.add_argument(
        "--save-model-interval",
        type=int,
        default=0,
        metavar="N",
        help="interval between saving model (default: 0, means don't save)",
    )
    parser.add_argument(
        "--gpu-index", 
        type=int, 
        default=0, 
        metavar="N"
    )
    parser.add_argument(
        "--optim-epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs for optimizer (default: 10)",
    )
    parser.add_argument(
        "--optim-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="batch size for optimizer (default: 64)",
    )
    return parser.parse_args()
