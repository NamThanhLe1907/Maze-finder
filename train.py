import threading
import time
import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pygame

from maze_env import MazeEnv, ComplexMazeEnv, SuperComplexMazeEnv
from dqn_agent import DQN, ReplayMemory

# Khởi tạo Lock để đồng bộ dữ liệu giữa training và GUI
state_lock = threading.Lock()

# --- CÁC HẰNG SỐ CHO GUI ---
CELL_SIZE = 50    # kích thước mỗi ô
MARGIN = 2        # khoảng cách giữa các ô
BTN_WIDTH = 100
BTN_HEIGHT = 40

# Màu sắc (RGB)
COLOR_EMPTY = (255, 255, 255)    # trắng: đường đi
COLOR_WALL = (0, 0, 0)           # đen: tường
COLOR_AGENT = (0, 255, 0)        # xanh: agent
COLOR_GOAL = (255, 0, 0)         # đỏ: đích
COLOR_BTN_BG = (100, 100, 255)   # nền nút
COLOR_BTN_TEXT = (255, 255, 255) # chữ nút

# --- BIẾN TOÀN CỤC CHIA SẺ GIỮA TRAIN VÀ GUI ---
simulation_data = {
    "epoch": 0,
    "state": None,            # Ma trận trạng thái hiện tại (agent được đánh dấu bằng 2)
    "action": None,           # Hành động dự đoán của agent tại trạng thái hiện tại
    "episode_finished": False,# Cờ báo kết thúc episode
    "training_done": False    # Cờ báo quá trình training hoàn thành
}

# --- HÀM TRAINING ---
def training_loop():
    global simulation_data
    # Sử dụng môi trường ComplexMazeEnv
    env = SuperComplexMazeEnv()
    obs_shape = env.maze.shape  # Với ComplexMazeEnv: (10,10)
    num_actions = env.action_space.n

    # Khởi tạo model; lưu ý: lớp fc trong DQN được định nghĩa là:
    # nn.Linear(input_size, 128) và nn.Linear(128, num_actions)
    # Với MazeEnv cũ: input_size = 25, còn với ComplexMazeEnv: input_size = 100.
    # Do đó, chúng ta không thể load toàn bộ trọng số. Ta sẽ load các tầng có kích thước khớp (ở đây, tầng cuối cùng).
    policy_net = DQN(obs_shape, num_actions)
    target_net = DQN(obs_shape, num_actions)
    target_net.load_state_dict(policy_net.state_dict())

    # Nếu file .pth tồn tại, ta sẽ cố load trọng số model cũ theo kiểu partial:
    try:
        loaded_state = torch.load("policy_net.pth", map_location=torch.device('cpu'))
        model_state = policy_net.state_dict()
        # Chỉ copy các trọng số có kích thước khớp
        for name, param in loaded_state.items():
            if name in model_state and model_state[name].size() == param.size():
                model_state[name].copy_(param)
        policy_net.load_state_dict(model_state)
        print("Partial weights loaded from policy_net.pth")
    except Exception as e:
        print("Could not load weights:", e)

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    memory = ReplayMemory(capacity=10000)

    num_episodes = 100
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update = 10

    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False

        # Lưu thời gian bắt đầu của episode
        episode_start_time = time.time()

        # Cập nhật dữ liệu ban đầu cho GUI
        with state_lock:
            simulation_data["epoch"] = episode
            simulation_data["state"] = state
            simulation_data["action"] = None
            simulation_data["episode_finished"] = False

        while not done:
            # Nếu vượt quá 10 giây, áp dụng hình phạt và kết thúc episode
            if time.time() - episode_start_time > 1000000:
                penalty = +0.1  # Hình phạt (có thể điều chỉnh)
                total_reward += penalty
                print(f"Episode {episode}: Timeout exceeded. Applying penalty: {penalty}")
                done = True
                break

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            memory.push((state_tensor, action, reward, next_state_tensor, done))
            state_tensor = next_state_tensor
            total_reward += reward

            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.cat(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards).unsqueeze(1)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + (gamma * max_next_q * (1 - dones))
                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Cập nhật dữ liệu chia sẻ cho GUI
            with state_lock:
                simulation_data["state"] = env._get_obs()
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(simulation_data["state"], dtype=torch.float32).unsqueeze(0))
                    simulation_data["action"] = q_vals.argmax().item()
            time.sleep(0.1)  # Delay mỗi bước (có thể điều chỉnh)

        rewards_per_episode.append(total_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}: Total Reward = {total_reward}")
        # Đánh dấu episode đã kết thúc và cho GUI hiển thị (giảm thời gian hiển thị, ví dụ 0.05 giây)
        with state_lock:
            simulation_data["episode_finished"] = True
        time.sleep(0.05)

        # Reset môi trường cho episode tiếp theo
        state = env.reset()
        with state_lock:
            simulation_data["state"] = state
            simulation_data["episode_finished"] = False

    simulation_data["training_done"] = True
    torch.save(policy_net.state_dict(), "policy_net.pth")
    # Vẽ đồ thị reward sau training (backend Agg nếu cần, để tránh lỗi GUI)



# --- HÀM GUI ---
def draw_maze(screen, maze, agent_pos, goal_pos):
    rows, cols = maze.shape
    for row in range(rows):
        for col in range(cols):
            color = COLOR_EMPTY if maze[row, col] == 0 else COLOR_WALL
            pygame.draw.rect(screen, color, [(MARGIN + CELL_SIZE) * col + MARGIN,
                                               (MARGIN + CELL_SIZE) * row + MARGIN,
                                               CELL_SIZE, CELL_SIZE])
    # Vẽ đích
    goal_rect = pygame.Rect((MARGIN + CELL_SIZE) * goal_pos[1] + MARGIN,
                              (MARGIN + CELL_SIZE) * goal_pos[0] + MARGIN,
                              CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, COLOR_GOAL, goal_rect)
    # Vẽ agent
    agent_rect = pygame.Rect((MARGIN + CELL_SIZE) * agent_pos[1] + MARGIN,
                              (MARGIN + CELL_SIZE) * agent_pos[0] + MARGIN,
                              CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, COLOR_AGENT, agent_rect)

def draw_button(screen, rect, text, font):
    pygame.draw.rect(screen, COLOR_BTN_BG, rect)
    text_surface = font.render(text, True, COLOR_BTN_TEXT)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def gui_loop():
    pygame.init()
    font = pygame.font.SysFont(None, 30)

    # Sử dụng ComplexMazeEnv để xác định kích thước GUI (đồng nhất với môi trường training)
    env = SuperComplexMazeEnv()
    maze = env.maze
    rows, cols = maze.shape
    window_width = cols * (CELL_SIZE + MARGIN) + MARGIN
    window_height = rows * (CELL_SIZE + MARGIN) + MARGIN + 100
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Interactive Maze RL GUI")

    # Vẽ nút (START và FINISH chỉ để hiển thị thông tin)
    btn_start_rect = pygame.Rect(10, window_height - 80, BTN_WIDTH, BTN_HEIGHT)
    btn_finish_rect = pygame.Rect(window_width - BTN_WIDTH - 10, window_height - 80, BTN_WIDTH, BTN_HEIGHT)

    agent_name = "Wins"
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Không xử lý nút bấm từ GUI vì reset được thực hiện trong training

        screen.fill((200, 200, 200))

        # Đọc dữ liệu chia sẻ từ simulation_data
        with state_lock:
            state = simulation_data.get("state")
            epoch = simulation_data.get("epoch", 0)
            best_action = simulation_data.get("action", None)
            episode_finished = simulation_data.get("episode_finished", False)
            training_done = simulation_data.get("training_done", False)
        if state is None:
            state = env._get_obs()
        agent_pos = tuple(np.argwhere(state == 2)[0])
        goal_pos = env.goal

        draw_maze(screen, maze, agent_pos, goal_pos)
        info_text = font.render(f"Agent: {agent_name} | Epoch: {epoch} | Action: {best_action}", True, (0,0,0))
        screen.blit(info_text, (10, window_height - 100))

        # Vẽ nút START và FINISH (chỉ hiển thị)
        draw_button(screen, btn_start_rect, "START", font)
        draw_button(screen, btn_finish_rect, "FINISH", font)

        if episode_finished:
            finish_text = font.render("Episode Finished. Resetting...", True, (255,0,0))
            screen.blit(finish_text, (window_width//2 - 100, 10))

        if training_done:
            done_text = font.render("Training Completed!", True, (0,255,0))
            screen.blit(done_text, (window_width//2 - 100, 40))
            pygame.display.flip()
            time.sleep(5)
            pygame.quit()
            sys.exit()

        pygame.display.flip()
        clock.tick(60)


# --- CHẠY SONG SONG TRAINING VÀ GUI ---
if __name__ == "__main__":
    train_thread = threading.Thread(target=training_loop)
    gui_thread = threading.Thread(target=gui_loop)

    train_thread.start()
    gui_thread.start()

    train_thread.join()
    gui_thread.join()
