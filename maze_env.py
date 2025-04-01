import gym
from gym import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        # Ma trận mê cung: 0 là đường đi, 1 là tường
        self.maze = np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0]
        ])
        self.start = (0, 0)
        self.goal = (4, 4)
        self.current_position = self.start

        # 4 hành động: 0 - lên, 1 - xuống, 2 - trái, 3 - phải
        self.action_space = spaces.Discrete(4)
        # Observation là ma trận mê cung (với agent được đánh dấu bằng số 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.maze.shape, dtype=np.uint8)

    def reset(self):
        self.current_position = self.start
        return self._get_obs()

    def _get_obs(self):
        obs = np.copy(self.maze)
        x, y = self.current_position
        obs[x, y] = 2  # đánh dấu vị trí agent
        return obs

    def step(self, action):
        x, y = self.current_position
        if action == 0:  # lên
            new_pos = (max(x - 1, 0), y)
        elif action == 1:  # xuống
            new_pos = (min(x + 1, self.maze.shape[0] - 1), y)
        elif action == 2:  # trái
            new_pos = (x, max(y - 1, 0))
        elif action == 3:  # phải
            new_pos = (x, min(y + 1, self.maze.shape[1] - 1))
        else:
            new_pos = (x, y)

        # Nếu new_pos là tường thì không di chuyển
        if self.maze[new_pos] == 1:
            new_pos = self.current_position

        self.current_position = new_pos
        done = (new_pos == self.goal)
        reward = 1 if done else -0.1  # thưởng khi đến đích, phạt nhẹ mỗi bước đi
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        # In ra ma trận mê cung
        print(self._get_obs())
        
class ComplexMazeEnv(MazeEnv):
    def __init__(self):
        # Ví dụ: ma trận 10x10 với nhiều tường hơn
        self.maze = np.array([
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
        ])
        self.start = (0, 0)
        self.goal = (4, 9)
        self.current_position = self.start
        # Cập nhật action_space và observation_space nếu cần
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.maze.shape, dtype=np.uint8)
        
    def reset(self):
        self.current_position = self.start
        return self._get_obs()

    def _get_obs(self):
        obs = np.copy(self.maze)
        x, y = self.current_position
        obs[x, y] = 2  # đánh dấu vị trí agent
        return obs

    def step(self, action):
        x, y = self.current_position
        if action == 0:  # lên
            new_pos = (max(x - 1, 0), y)
        elif action == 1:  # xuống
            new_pos = (min(x + 1, self.maze.shape[0] - 1), y)
        elif action == 2:  # trái
            new_pos = (x, max(y - 1, 0))
        elif action == 3:  # phải
            new_pos = (x, min(y + 1, self.maze.shape[1] - 1))
        else:
            new_pos = (x, y)

        # Nếu new_pos là tường thì không di chuyển
        if self.maze[new_pos] == 1:
            new_pos = self.current_position

        self.current_position = new_pos
        done = (new_pos == self.goal)
        reward = 1 if done else -0.1  # thưởng khi đến đích, phạt nhẹ mỗi bước đi
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        # In ra ma trận mê cung
        print(self._get_obs())
        
import gym
from gym import spaces
import numpy as np

class SuperComplexMazeEnv(gym.Env):
    def __init__(self):
        super(SuperComplexMazeEnv, self).__init__()
        # Ví dụ: Ma trận 15x15
        self.maze = np.array([
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]
        ])
        self.start = (0, 0)
        self.goal = (14, 7)
        self.current_position = self.start

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2, shape=self.maze.shape, dtype=np.uint8)

    def reset(self):
        self.current_position = self.start
        return self._get_obs()

    def _get_obs(self):
        obs = np.copy(self.maze)
        x, y = self.current_position
        obs[x, y] = 2  # đánh dấu vị trí agent
        return obs

    def step(self, action):
        x, y = self.current_position
        if action == 0:
            new_pos = (max(x - 1, 0), y)
        elif action == 1:
            new_pos = (min(x + 1, self.maze.shape[0] - 1), y)
        elif action == 2:
            new_pos = (x, max(y - 1, 0))
        elif action == 3:
            new_pos = (x, min(y + 1, self.maze.shape[1] - 1))
        else:
            new_pos = (x, y)

        if self.maze[new_pos] == 1:
            new_pos = self.current_position

        self.current_position = new_pos
        done = (new_pos == self.goal)
        
        # Tính khoảng cách tới đích
        distance = np.linalg.norm(np.array(new_pos) - np.array(self.goal))
        
        # Tính khoảng cách mới và cũ
        old_distance = np.linalg.norm(np.array(self.current_position) - np.array(self.goal))
        new_distance = distance
        
        # Reward cơ bản
        reward = -0.01  # Phạt nhẹ mỗi bước
        
        # Thưởng/phạt dựa trên hướng di chuyển
        if new_distance < old_distance:
            reward += 0.1  # Thưởng khi tiến gần đích
        elif new_distance > old_distance:
            reward -= 0.05  # Phạt nhẹ khi đi xa đích
            
        # Thêm reward dựa trên khoảng cách
        reward += 0.3 / (new_distance + 1)  # +1 để tránh chia cho 0
        
        # Thêm bonus khi đạt mục tiêu
        if done:
            reward += 10.0  # Bonus lớn khi thành công
            
        # Thông tin bổ sung
        info = {
            'distance': distance,
            'success': done,
            'position': new_pos
        }
        
        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        print(self._get_obs())
