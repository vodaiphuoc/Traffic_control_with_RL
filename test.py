from sumo_rl import SumoEnvironment
import torch

device = torch.device("cpu")

env = SumoEnvironment(
    net_file="sumo_rl_repo/nets/2way-single-intersection/single-intersection.net.xml",
    route_file="sumo_rl_repo/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    out_csv_name="outputs/2way-single-intersection/dqn",
    single_agent=True,
    use_gui=True,
    num_seconds=100000,
)

state, info = env.reset()

# observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
observation, reward, terminated, truncated, infor = env.step(action= 2)
