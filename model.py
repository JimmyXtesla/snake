import pygame
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- CONFIGURATION ---
GRID_SIZE = 30
GRID_COUNT = 14  # 14x14 arena
WINDOW_SIZE = GRID_SIZE * GRID_COUNT

# Auto-detect local acceleration architecture
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  
else:
    DEVICE = torch.device("cpu")

COLOR_BG = (15, 18, 26)
COLOR_WALL = (33, 43, 54)
COLOR_HEAD = (52, 152, 219)
COLOR_BODY = (41, 128, 185)
COLOR_APPLE = (231, 76, 60)
COLOR_TEXT = (236, 240, 241)


class WorldClassSnakeGame:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("PER-DDDQN Accelerated Snake")
        self.clock = pygame.time.Clock()
        self.generate_hamiltonian_cycle()
        self.reset()

    def generate_hamiltonian_cycle(self):
        """Generates a perfect fixed grid cycle covering every cell exactly once."""
        self.cycle_idx = {}
        path = []
        for x in range(GRID_COUNT):
            path.append((0, x))
        for y in range(1, GRID_COUNT):
            if y % 2 == 1:
                for x in reversed(range(1, GRID_COUNT)):
                    path.append((y, x))
            else:
                for x in range(1, GRID_COUNT):
                    path.append((y, x))
        for y in reversed(range(1, GRID_COUNT)):
            path.append((y, 0))
            
        for idx, pos in enumerate(path):
            self.cycle_idx[pos] = idx

    def reset(self):
        self.snake = [
            (GRID_COUNT // 2, GRID_COUNT // 2),
            (GRID_COUNT // 2, GRID_COUNT // 2 - 1),
            (GRID_COUNT // 2, GRID_COUNT // 2 - 2)
        ]
        self.snake_set = set(self.snake[:-1])  # O(1) performance lookup mapping
        self.direction = 1  # 0: Up, 1: Right, 2: Down, 3: Left
        self.score = 0
        self.frame_iteration = 0
        self._place_apple()
        return self.get_radar_state()

    def _place_apple(self):
        total_cells = GRID_COUNT * GRID_COUNT
        if len(self.snake) >= total_cells:
            return
        snake_full_set = set(self.snake)
        while True:
            self.apple = (random.randint(0, GRID_COUNT - 1), random.randint(0, GRID_COUNT - 1))
            if self.apple not in snake_full_set:
                break

    def get_radar_state(self):
        """Casts 8 directional vision vectors for high-dimensional spatial awareness."""
        head_y, head_x = self.snake[0]
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        state = []

        for dy, dx in directions:
            wall_dist, body_found, apple_found = 0.0, 0.0, 0.0
            cy, cx = head_y, head_x
            dist = 0
            while True:
                cy += dy; cx += dx; dist += 1
                if cx < 0 or cx >= GRID_COUNT or cy < 0 or cy >= GRID_COUNT:
                    wall_dist = 1.0 / dist
                    break
                if (cy, cx) in self.snake_set and body_found == 0.0:
                    body_found = 1.0 / dist
                if (cy, cx) == self.apple:
                    apple_found = 1.0
            state.extend([wall_dist, body_found, apple_found])
        return np.array(state, dtype=np.float32)

    def step(self, action, use_safety=True):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        proposed_dir = self.direction
        if action == 1:   # Left
            proposed_dir = (self.direction - 1) % 4
        elif action == 2: # Right
            proposed_dir = (self.direction + 1) % 4

        dy = [-1, 0, 1, 0]; dx = [0, 1, 0, -1]
        head_y, head_x = self.snake[0]
        
        if use_safety:
            best_action_dir = proposed_dir
            min_distance_to_apple = float('inf')
            valid_safe_move_found = False
            backup_hamiltonian_dir = proposed_dir
            
            for look_dir in range(4):
                ny, nx = head_y + dy[look_dir], head_x + dx[look_dir]
                if 0 <= nx < GRID_COUNT and 0 <= ny < GRID_COUNT and (ny, nx) not in self.snake_set:
                    current_idx = self.cycle_idx[(head_y, head_x)]
                    next_idx = self.cycle_idx[(ny, nx)]
                    tail_idx = self.cycle_idx[self.snake[-1]]
                    
                    if next_idx == (current_idx + 1) % (GRID_COUNT * GRID_COUNT):
                        backup_hamiltonian_dir = look_dir 
                    
                    dist_to_tail = (tail_idx - next_idx) % (GRID_COUNT * GRID_COUNT)
                    if dist_to_tail > 1:
                        dist_to_apple = abs(ny - self.apple[0]) + abs(nx - self.apple[1])
                        if dist_to_apple < min_distance_to_apple:
                            min_distance_to_apple = dist_to_apple
                            best_action_dir = look_dir
                            valid_safe_move_found = True
            
            if valid_safe_move_found:
                self.direction = best_action_dir
            else:
                self.direction = backup_hamiltonian_dir

        new_head = (head_y + dy[self.direction], head_x + dx[self.direction])
        
        if (new_head[1] < 0 or new_head[1] >= GRID_COUNT or 
            new_head[0] < 0 or new_head[0] >= GRID_COUNT or 
            new_head in self.snake_set):
            return self.get_radar_state(), -10.0, True

        self.snake.insert(0, new_head)
        if new_head == self.apple:
            self.score += 1
            reward = 10.0
            self._place_apple()
        else:
            reward = 0.01
            self.snake.pop()

        self.snake_set = set(self.snake[:-1])
        return self.get_radar_state(), reward, False

    def render(self, score_history):
        self.window.fill(COLOR_BG)
        pygame.draw.rect(self.window, COLOR_WALL, (0, 0, WINDOW_SIZE, WINDOW_SIZE), 3)
        
        pygame.draw.rect(self.window, COLOR_APPLE, (self.apple[1]*GRID_SIZE+2, self.apple[0]*GRID_SIZE+2, GRID_SIZE-4, GRID_SIZE-4), border_radius=6)
        for idx, (y, x) in enumerate(self.snake):
            color = COLOR_HEAD if idx == 0 else COLOR_BODY
            pygame.draw.rect(self.window, color, (x*GRID_SIZE+1, y*GRID_SIZE+1, GRID_SIZE-2, GRID_SIZE-2), border_radius=4)
            
        font = pygame.font.SysFont('Arial', 18, bold=True)
        max_score = max(score_history) if score_history else 0
        txt_stats = font.render(f"Current Score: {self.score}  |  All-Time Record: {max_score}", True, COLOR_TEXT)
        self.window.blit(txt_stats, (15, 15))
        pygame.display.flip()


# --- PRIORITIZED EXPERIENCE REPLAY BUFFER (PER) ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def append(self, state, action, reward, next_state, done):
        max_p = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


# --- DUELING DEEP Q-NETWORK BRAIN ---
class DuelingBrain(nn.Module):
    def __init__(self, input_dim=24, output_dim=3):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=-1, keepdim=True))


if __name__ == "__main__":
    print(f"Executing deep learning sequence locally using framework target: {DEVICE}")
    game = WorldClassSnakeGame()
    policy_net = DuelingBrain().to(DEVICE)
    target_net = DuelingBrain().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=0.0003, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == "cuda" else None
    
    memory = PrioritizedReplayBuffer(capacity=50000)
    score_history = []
    
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.992
    epsilon_min = 0.01
    generation = 0
    beta_start = 0.4

    while True:
        state = game.reset()
        done = False
        generation += 1
        beta = min(1.0, beta_start + generation * 0.002)
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
                    action = policy_net(state_t).argmax().item()
                    
            next_state, reward, done = game.step(action, use_safety=True)
            memory.append(state, action, reward, next_state, done)
            state = next_state
            
            if len(memory) > batch_size:
                samples, indices, weights = memory.sample(batch_size, beta=beta)
                s, a, r, ns, d = zip(*samples)
                
                s_t = torch.FloatTensor(np.array(s)).to(DEVICE)
                a_t = torch.LongTensor(a).to(DEVICE).unsqueeze(1)
                r_t = torch.FloatTensor(r).to(DEVICE)
                ns_t = torch.FloatTensor(np.array(ns)).to(DEVICE)
                d_t = torch.FloatTensor(d).to(DEVICE)
                w_t = torch.FloatTensor(weights).to(DEVICE)
                
                # Dynamic optimization runtime path context mapping
                if scaler and DEVICE.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        current_q = policy_net(s_t).gather(1, a_t).squeeze()
                        next_actions = policy_net(ns_t).argmax(dim=1, keepdim=True)
                        next_q = target_net(ns_t).gather(1, next_actions).squeeze().detach()
                        target_q = r_t + (gamma * next_q * (1 - d_t))
                        
                        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
                        loss = (w_t * nn.HuberLoss(reduction='none')(current_q, target_q)).mean()
                        
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    current_q = policy_net(s_t).gather(1, a_t).squeeze()
                    next_actions = policy_net(ns_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(ns_t).gather(1, next_actions).squeeze().detach()
                    target_q = r_t + (gamma * next_q * (1 - d_t))
                    
                    td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
                    loss = (w_t * nn.HuberLoss(reduction='none')(current_q, target_q)).mean()
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    optimizer.step()
                
                memory.update_priorities(indices, td_errors + 1e-5)
                
            if generation > 15:
                game.render(score_history)
                game.clock.tick(120)  # Smooth running on local high refresh monitors
                
        score_history.append(game.score)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Soft Polyak Target Net Sync (Prevents oscillations during local execution)
        tau = 0.05
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
            
        if generation % 50 == 0:
            torch.save(policy_net.state_dict(), "snake_brain.pth")
            print(f"--> [Gen {generation}] Saved local training checkpoint. Record Score: {max(score_history)}")
