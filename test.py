# from src.environment.env import SumoEnvironment

# env = SumoEnvironment(
#     net_file="src/nets/2way-single-intersection/single-intersection.net.xml",
#     route_file="src/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
#     out_csv_name="outputs/2way-single-intersection/dqn",
#     single_agent=True,
#     use_gui=True,
#     num_seconds=100000,
# )


# state, info = env.reset()


# observation, reward, terminated, truncated, infor = env.step(2)

from itertools import count
for i in range(10):
    for t in count():
        print(t)