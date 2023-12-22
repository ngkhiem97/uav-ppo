import sys
from utils import *
from core.agent import Agent
from core.env import UnderwaterNavigation

# Append system path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Parse arguments
args = parse_arguments()

# Set default data type and device for torch
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

# Initialize environments and agent
start_goal_pos = [15.0, -2.5, -15, 0.0, 270.0, 0.0, 5, -1.5, -5]
environments = [UnderwaterNavigation(args.depth_prediction_model, args.adaptation, args.randomization, i, args.hist_length, start_goal_pos) for i in range(args.num_threads)]

# Initialize networks
policy_net = Policy(args.hist_length, environments[0].action_space.shape[0], log_std=args.log_std)
policy_net.load_state_dict(torch.load("policy.pth"))
policy_net.eval()
policy_net.to(device)

with open("running_state.p", "rb") as file:
    running_state = pickle.load(file)

# Initialize agent
agent = Agent(
    environments,
    policy_net,
    device,
    running_state=running_state,
    num_threads=args.num_threads,
    training=False,
)

# Evaluate agent
while True:
    if args.eval_batch_size > 0:
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)