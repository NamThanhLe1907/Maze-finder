import pygame
import sys
import time
import torch
import numpy as np
from maze_env import MazeEnv
from dqn_agent import DQN

# Cấu hình giao diện
CELL_SIZE = 50    # kích thước mỗi ô
MARGIN = 2        # khoảng cách giữa các ô

# Định nghĩa màu sắc
COLOR_EMPTY = (255, 255, 255)    # trắng: đường đi
COLOR_WALL = (0, 0, 0)           # đen: tường
COLOR_AGENT = (0, 255, 0)        # xanh: agent
COLOR_GOAL = (255, 0, 0)         # đỏ: đích
COLOR_BTN_BG = (100, 100, 255)   # nền nút
COLOR_BTN_TEXT = (255, 255, 255) # chữ nút

# Kích thước nút
BTN_WIDTH = 100
BTN_HEIGHT = 40

def draw_maze(screen, maze, agent_pos, goal_pos):
    rows, cols = maze.shape
    for row in range(rows):
        for col in range(cols):
            color = COLOR_EMPTY if maze[row, col] == 0 else COLOR_WALL
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + CELL_SIZE) * col + MARGIN,
                              (MARGIN + CELL_SIZE) * row + MARGIN,
                              CELL_SIZE,
                              CELL_SIZE])
    # Vẽ đích
    goal_rect = pygame.Rect((MARGIN + CELL_SIZE) * goal_pos[1] + MARGIN,
                            (MARGIN + CELL_SIZE) * goal_pos[0] + MARGIN,
                            CELL_SIZE,
                            CELL_SIZE)
    pygame.draw.rect(screen, COLOR_GOAL, goal_rect)
    
    # Vẽ agent
    agent_rect = pygame.Rect((MARGIN + CELL_SIZE) * agent_pos[1] + MARGIN,
                             (MARGIN + CELL_SIZE) * agent_pos[0] + MARGIN,
                             CELL_SIZE,
                             CELL_SIZE)
    pygame.draw.rect(screen, COLOR_AGENT, agent_rect)

def draw_button(screen, rect, text, font):
    pygame.draw.rect(screen, COLOR_BTN_BG, rect)
    text_surface = font.render(text, True, COLOR_BTN_TEXT)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def interactive_gui(env, policy_net, delay=0.5, run_time=15):
    """
    Chạy giao diện GUI tương tác để hiển thị quá trình di chuyển của agent.
    
    Parameters:
      env: môi trường MazeEnv (đã được reset)
      policy_net: model đã được huấn luyện (DQN)
      delay: thời gian chờ giữa các bước (giây)
      run_time: thời gian chạy simulation (giây)
    """
    pygame.init()
    font = pygame.font.SysFont(None, 30)
    
    maze = env.maze
    rows, cols = maze.shape
    window_width = cols * (CELL_SIZE + MARGIN) + MARGIN
    window_height = rows * (CELL_SIZE + MARGIN) + MARGIN + 100  # không gian cho nút & info
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Interactive Maze RL GUI")
    
    # Định nghĩa nút START và FINISH
    btn_start_rect = pygame.Rect(10, window_height - 80, BTN_WIDTH, BTN_HEIGHT)
    btn_finish_rect = pygame.Rect(window_width - BTN_WIDTH - 10, window_height - 80, BTN_WIDTH, BTN_HEIGHT)
    
    agent_name = "DQN Agent"
    
    clock = pygame.time.Clock()
    running = True
    simulation_running = False  # trạng thái simulation
    start_time = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if btn_start_rect.collidepoint(mouse_pos) and not simulation_running:
                    simulation_running = True
                    start_time = time.time()
                    env.reset()  # reset môi trường
                elif btn_finish_rect.collidepoint(mouse_pos) and simulation_running:
                    simulation_running = False
        
        screen.fill((200, 200, 200))
        
        # Lấy trạng thái hiện tại của môi trường
        current_obs = env._get_obs()
        agent_pos = tuple(np.argwhere(current_obs == 2)[0])
        goal_pos = env.goal
        
        # Nếu simulation đang chạy, tiến hành simulation
        if simulation_running:
            # Dự đoán hành động từ model
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(obs_tensor)
                best_action = q_values.argmax().item()
            # Cập nhật môi trường
            next_obs, reward, done, _ = env.step(best_action)
            if done:
                simulation_running = False
            # Vẽ mê cung
            draw_maze(screen, env.maze, agent_pos, goal_pos)
            # Hiển thị thông tin
            info_text = font.render(f"Agent: {agent_name} | Action: {best_action}", True, (0, 0, 0))
            screen.blit(info_text, (10, window_height - 100))
            time.sleep(delay)
        else:
            # Vẽ trạng thái tĩnh khi simulation không chạy
            draw_maze(screen, env.maze, agent_pos, goal_pos)
            info_text = font.render(f"Agent: {agent_name}", True, (0, 0, 0))
            screen.blit(info_text, (10, window_height - 100))
        
        # Hiển thị nút
        draw_button(screen, btn_start_rect, "START", font)
        draw_button(screen, btn_finish_rect, "FINISH", font)
        
        # Nếu simulation đang chạy, kiểm tra thời gian
        if simulation_running and start_time is not None:
            elapsed = time.time() - start_time
            if elapsed > run_time:
                simulation_running = False
        # Hiển thị timer
        timer_text = font.render(f"Time: {int(time.time() - start_time) if start_time else 0}s", True, (0, 0, 255))
        screen.blit(timer_text, (window_width - 150, window_height - 100))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    # Để chạy GUI độc lập, ví dụ:
    env = MazeEnv()
    env.reset()
    # Khởi tạo model mẫu
    obs_shape = env.maze.shape
    num_actions = env.action_space.n
    policy_net = DQN(obs_shape, num_actions)
    # Nếu bạn có file model đã lưu, bạn có thể load nó ở đây.
    interactive_gui(env, policy_net, delay=0.5, run_time=15)
