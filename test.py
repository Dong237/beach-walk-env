from beach_walk_env.beach_walk import BeachWalkEnv
from tqdm import tqdm

env = BeachWalkEnv(
    size=6, 
    agent_start_pos=(1, 2), 
    agent_start_dir=0, 
    max_steps=25, 
    wind_gust_probability=0.5,
    reward=1., 
    penalty=-1., 
    discount=.95,
)

observation, info = env.reset()

for _ in tqdm(range(500000)):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()