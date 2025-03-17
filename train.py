import threading
import time
import sys
import torch
torch.set_default_dtype(torch.float32)
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pygame

from maze_env import SuperComplexMazeEnv
from dqn_agent import DQN, PrioritizedReplayMemory

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
    "training_done": False,   # Cờ báo quá trình training hoàn thành
    "steps_taken": 0,         # Số ô đã di chuyển trong mỗi episode
    "speed": 0.001,           # Thời gian sleep (giây) được điều khiển bởi slider
}

# --- HÀM TRAINING ---
def training_loop():
    global simulation_data
    env = SuperComplexMazeEnv()
    obs_shape = env.maze.shape  # Với ComplexMazeEnv: (10,10)
    num_actions = env.action_space.n

    policy_net = DQN(obs_shape, num_actions)  # Initialize your model
    try:
        policy_net.load_state_dict(torch.load("policy_net.pth", map_location=torch.device('cpu')))  # Load weights
        policy_net.eval()  # Set the model to evaluation mode
        print("Model loaded successfully from policy_net.pth")
    except Exception as e:
        print("Could not load model weights:", e)
    target_net = DQN(obs_shape, num_actions)
    target_net.load_state_dict(policy_net.state_dict())

    try:
        loaded_state = torch.load("policy_net.pth", map_location=torch.device('cpu'))
        model_state = policy_net.state_dict()
        for name, param in loaded_state.items():
            if name in model_state and model_state[name].size() == param.size():
                model_state[name].copy_(param)
        policy_net.load_state_dict(model_state)
        print("Partial weights loaded from policy_net.pth")
    except Exception as e:
        print("Could not load weights:", e)

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    memory = PrioritizedReplayMemory(capacity=10000)

    num_episodes = 500
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    target_update = 10

    rewards_per_episode = []
    performance_metrics = []  # To track performance across epochs

    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        done = False

        episode_start_time = time.time()

        with state_lock:
            simulation_data["epoch"] = episode
            simulation_data["state"] = state
            simulation_data["action"] = None
            simulation_data["episode_finished"] = False

        step_count = 0
        steps_taken = 0
        max_steps = 100
        while not done:
            step_count += 1
            
            if steps_taken >= max_steps:
                agent_pos = np.argwhere(state == 2)[0]
                goal_pos = env.goal
                distance = np.linalg.norm(agent_pos - goal_pos)
                penalty = -0.1 * distance
                total_reward += penalty
                distance = np.linalg.norm(agent_pos - goal_pos)
                total_reward += -0.1 * distance
                steps_taken += 1
                print(f"Episode {episode}: Max steps exceeded. Distance: {distance:.1f}, Penalty: {penalty:.2f}")
                done = True
                break

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            prev_pos = tuple(np.argwhere(state == 2)[0])
            
            next_state, reward, done, _ = env.step(action)
            current_pos = tuple(np.argwhere(next_state == 2)[0])
            if current_pos != prev_pos:
                steps_taken += 1
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            memory.push((state_tensor, action, reward, next_state_tensor, done))
            state = next_state
            state_tensor = next_state_tensor
            total_reward += reward

            if len(memory) >= batch_size:
                batch, indices, weights = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                weights = torch.tensor(weights, dtype=torch.float32)
                states = torch.cat(states).float()
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.cat(next_states).float()
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + (gamma * max_next_q * (1 - dones))
                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with state_lock:
                simulation_data["state"] = env._get_obs()
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(simulation_data["state"], dtype=torch.float32).unsqueeze(0))
                    simulation_data["action"] = q_vals.argmax().item()
                simulation_data["steps_taken"] = steps_taken
            time.sleep(simulation_data["speed"])  # Use direct speed value from slider in seconds

        rewards_per_episode.append(total_reward)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(0.005 * param.data + (1.0 - 0.005) * target_param.data)

        print(f"Episode {episode}: Total Reward = {total_reward}")
        with state_lock:
            simulation_data["episode_finished"] = True
        time.sleep(0.05)

        state = env.reset()
        with state_lock:
            simulation_data["state"] = state
            simulation_data["episode_finished"] = False

    simulation_data["training_done"] = True
    torch.save(policy_net.state_dict(), "policy_net.pth")

def draw_maze(screen, maze, agent_pos, goal_pos):
    rows, cols = maze.shape
    for row in range(rows):
        for col in range(cols):
            color = COLOR_EMPTY if maze[row, col] == 0 else COLOR_WALL
            rect = [
                (MARGIN + CELL_SIZE) * col + MARGIN,
                (MARGIN + CELL_SIZE) * row + MARGIN,
                CELL_SIZE, CELL_SIZE
            ]
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200,200,200), rect, 1)
    goal_rect = pygame.Rect(
        (MARGIN + CELL_SIZE) * goal_pos[1] + MARGIN,
        (MARGIN + CELL_SIZE) * goal_pos[0] + MARGIN,
        CELL_SIZE, CELL_SIZE
    )
    pygame.draw.rect(screen, COLOR_GOAL, goal_rect)
    pygame.draw.rect(screen, (0,0,0), goal_rect, 2)
    agent_rect = pygame.Rect(
        (MARGIN + CELL_SIZE) * agent_pos[1] + MARGIN + 2,
        (MARGIN + CELL_SIZE) * agent_pos[0] + MARGIN + 2,
        CELL_SIZE-4, CELL_SIZE-4
    )
    pygame.draw.rect(screen, (0,200,0), agent_rect.inflate(4,4))
    pygame.draw.rect(screen, COLOR_AGENT, agent_rect)
    pygame.draw.rect(screen, (0,100,0), agent_rect, 2)

def draw_slider(screen, rect, value, font, text="Adjust Speed"):
    pygame.draw.rect(screen, (200, 200, 200), rect)
    pygame.draw.rect(screen, (0, 0, 0), (rect.x, rect.y, rect.width, 20))
    pygame.draw.rect(screen, (0, 255, 0), (rect.x, rect.y, value, 20))
    speed_display = f"{(200 - value)/1000.0:.2f}s"  # Hiển thị thời gian sleep thực tế
    text_surface = font.render(f"Speed: {speed_display}", True, (0, 0, 0))
    screen.blit(text_surface, (rect.x + 5, rect.y - 20))

def draw_button(screen, rect, text, font):
    pygame.draw.rect(screen, COLOR_BTN_BG, rect)
    text_surface = font.render(text, True, COLOR_BTN_TEXT)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def gui_loop(env, policy_net, delay=0.5, run_time=15):
    pygame.display.init()
    pygame.font.init()
    
    maze = env.maze
    rows, cols = maze.shape
    window_width = cols * (CELL_SIZE + MARGIN) + MARGIN
    window_height = rows * (CELL_SIZE + MARGIN) + MARGIN + 100
    screen = pygame.display.set_mode(
        (window_width, window_height),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Interactive Maze RL GUI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 30, bold=True)
    
    # For slider UI (optional, currently not used for changing training speed)
    slider_rect = pygame.Rect(10, window_height - 50, 200, 20)
    slider_value = 100

    # Define button rectangles (display-only)
    btn_start_rect = pygame.Rect(10, window_height - 80, BTN_WIDTH, BTN_HEIGHT)
    btn_finish_rect = pygame.Rect(window_width - BTN_WIDTH - 10, window_height - 80, BTN_WIDTH, BTN_HEIGHT)
    agent_name = "Wins"

    running = True
    # Single main loop for GUI updates
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if slider_rect.collidepoint(event.pos):
                    slider_value = event.pos[0] - slider_rect.x
                    slider_value = max(0, min(slider_value, slider_rect.width))
                    with state_lock:
                        simulation_data["speed"] = (200 - slider_value) / 1000.0
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0] and slider_rect.collidepoint(event.pos):
                    slider_value = event.pos[0] - slider_rect.x
                    slider_value = max(0, min(slider_value, slider_rect.width))
                    with state_lock:
                        simulation_data["speed"] = (200 - slider_value) / 1000.0
        screen.fill((200, 200, 200))
        # Use updated state from training loop
        with state_lock:
            state = simulation_data.get("state")
            epoch = simulation_data.get("epoch", 0)
            best_action = simulation_data.get("action", None)
            episode_finished = simulation_data.get("episode_finished", False)
            training_done = simulation_data.get("training_done", False)
            steps_taken = simulation_data.get("steps_taken", 0)
        if state is None:
            state = env._get_obs()
        try:
            agent_pos = tuple(np.argwhere(state == 2)[0])
        except Exception:
            agent_pos = (0, 0)
        goal_pos = env.goal
        
        draw_maze(screen, maze, agent_pos, goal_pos)
        draw_slider(screen, slider_rect, slider_value, font)
        info_text = font.render(f"Agent: {agent_name} | Epoch: {epoch} | Action: {best_action} | Steps: {steps_taken}", True, (0,0,0))
        screen.blit(info_text, (10, window_height - 100))
        # Display buttons (if needed)
        # draw_button(screen, btn_start_rect, "START", font)
        # draw_button(screen, btn_finish_rect, "FINISH", font)
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

    pygame.quit()

# --- CHẠY SONG SONG TRAINING VÀ GUI ---
if __name__ == "__main__":
    env = SuperComplexMazeEnv()
    obs_shape = env.maze.shape
    num_actions = env.action_space.n
    policy_net = DQN(obs_shape, num_actions)

    train_thread = threading.Thread(target=training_loop)
    train_thread.start()

    gui_loop(env, policy_net)

    train_thread.join()
    pygame.quit()
