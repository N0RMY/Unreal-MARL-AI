import math
import os
import socket
import struct
import json
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import logging 

# --- ФІКС ШЛЯХІВ: Визначаємо точну папку, де лежить server.py ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIDER_MODEL_PATH = os.path.join(SCRIPT_DIR, "hider_model.pth")
SEEKER_MODEL_PATH = os.path.join(SCRIPT_DIR, "seeker_model.pth")
LOG_PATH = os.path.join(SCRIPT_DIR, "training_log.txt")

# ==========================================
# НАЛАШТУВАННЯ ЛОГУВАННЯ
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"), # Тепер лог теж зберігається в правильну папку
        logging.StreamHandler()
    ]
)

# ==========================================
# 1. Архітектура Нейромережі
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_head = nn.Linear(128, action_dim)
        
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.8))
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean_head(x))
        std = torch.exp(self.log_std)
        return mean, std

# ==========================================
# 2. Ініціалізація та Завантаження Пам'яті
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hider_model = PolicyNetwork(input_dim=16, action_dim=2).to(device)
seeker_model = PolicyNetwork(input_dim=16, action_dim=2).to(device)

# --- ЗМІНЕНО: Використовуємо абсолютні шляхи ---
if os.path.exists(HIDER_MODEL_PATH):
    try:
        hider_model.load_state_dict(torch.load(HIDER_MODEL_PATH, map_location=device, weights_only=True))
        logging.info("✅ Завантажено попередній мозок Hider-а!")
    except Exception as e:
        logging.warning(f"⚠️ Не вдалось завантажити Hider модель (несумісна архітектура?): {e}. Починаємо з нуля.")

if os.path.exists(SEEKER_MODEL_PATH):
    try:
        seeker_model.load_state_dict(torch.load(SEEKER_MODEL_PATH, map_location=device, weights_only=True))
        logging.info("✅ Завантажено попередній мозок Seeker-а!")
    except Exception as e:
        logging.warning(f"⚠️ Не вдалось завантажити Seeker модель (несумісна архітектура?): {e}. Починаємо з нуля.")

hider_model.train()
hider_optimizer = optim.Adam(hider_model.parameters(), lr=0.0005)

seeker_model.train()
seeker_optimizer = optim.Adam(seeker_model.parameters(), lr=0.0005)

MAX_STEPS_PER_EPISODE = 2000  # ~67 сек при 30 fps; страховка від OOM

class RolloutBuffer:
    def __init__(self):
        self.states  = []   # List[Tensor [16]], detached, CPU — НЕ зберігаємо граф
        self.actions = []   # List[Tensor [2]],  detached, CPU
        self.rewards = []   # List[float]

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

hider_memory = RolloutBuffer()
seeker_memory = RolloutBuffer()

# ==========================================
# 4. Логіка Тренування
# ==========================================
def train_step(model, optimizer, memory, agent_name):
    n = min(len(memory.states), len(memory.rewards))
    if n == 0:
        memory.clear()
        return

    total_reward = sum(memory.rewards[:n])
    steps_taken  = n

    # Дисконтовані повернення
    running_reward = 0
    returns = []
    gamma = 0.99
    for r in memory.rewards[:n][::-1]:
        running_reward = r + gamma * running_reward
        returns.insert(0, running_reward)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    else:
        returns = returns - returns.mean()

    # Батчевий форвард-пас: перераховуємо log_prob та entropy БЕЗ накопичення графів
    states_batch  = torch.stack(memory.states[:n]).to(device)   # [N, 16]
    actions_batch = torch.stack(memory.actions[:n]).to(device)  # [N, 2]

    mean, std = model(states_batch)
    dist      = Normal(mean, std)
    log_probs = dist.log_prob(actions_batch).sum(dim=-1)  # [N]
    entropies = dist.entropy().sum(dim=-1)                # [N]

    optimizer.zero_grad()
    loss = (-log_probs * returns).mean() - 0.01 * entropies.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    memory.clear()
    logging.info(f"[НАВЧАННЯ] {agent_name} оновлено! Loss: {loss.item():.4f} | Балів за раунд: {total_reward:.4f} | Кроків: {steps_taken}")

# ==========================================
# 5. Мережа та Обробка Даних
# ==========================================
HOST = "127.0.0.1"
PORT = 5555

MAX_POS = 7000.0  # Мапа 1: X[-2000..2000], Y[2680..6690]; Мапа 2: X[-2000..2000], Y[-2000..2000]
MAX_VEL = 1000.0
MAX_SENSOR = 1500.0

def recvall(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk: return None
        data += chunk
    return data

def handle_client(conn, addr):
    logging.info(f"Client connected: {addr}")
    last_dist = None 
    rounds_played = 0 
    
    # Змінна для відстеження попередньої дії (щоб штрафувати за задній хід)
    last_seeker_move = 0.0 
    
    with conn:
        while True:
            raw_len = recvall(conn, 4)
            if raw_len is None: return
            (msg_len,) = struct.unpack("!I", raw_len)

            if msg_len == 0:
                conn.sendall(struct.pack("!I", 14) + b'{"ok": False}')
                continue

            raw = recvall(conn, msg_len)
            if raw is None: return

            try:
                msg = json.loads(raw.decode("utf-8"))
                obs = msg.get("obs")
                if obs is None: continue 

                is_done = msg.get("done", False)

                h_x, h_y = obs["hider"][0], obs["hider"][1]
                s_x, s_y = obs["seeker"][0], obs["seeker"][1]
                dx = h_x - s_x
                dy = h_y - s_y
                dist = math.hypot(dx, dy) / MAX_POS
                
                target_angle = math.atan2(dy, dx)
                s_yaw_rad = math.radians(obs.get("seeker_yaw", 0.0))
                s_rel_angle = target_angle - s_yaw_rad
                s_rel_angle = (s_rel_angle + math.pi) % (2 * math.pi) - math.pi
                seeker_radar = s_rel_angle / math.pi
                
                h_target_angle = math.atan2(-dy, -dx)
                h_yaw_rad = math.radians(obs.get("hider_yaw", 0.0))
                h_rel_angle = h_target_angle - h_yaw_rad
                h_rel_angle = (h_rel_angle + math.pi) % (2 * math.pi) - math.pi
                hider_radar = h_rel_angle / math.pi

                # Читаємо сенсори ДО розрахунку нагород
                h_sensors = [s / MAX_SENSOR for s in obs.get("hider_sensors", [0.0] * 8)]
                s_sensors = [s / MAX_SENSOR for s in obs.get("seeker_sensors", [0.0] * 8)]

                # ====================================================
                # СИСТЕМА НАГОРОД
                # ====================================================
                seeker_reward = 0.0
                
                if last_dist is not None:
                    dist_delta = last_dist - dist
                    seeker_reward += dist_delta * 50.0 
                
                last_dist = dist
                
                facing_bonus = 1.0 - abs(seeker_radar)
                seeker_reward += facing_bonus * 0.01 
                
                seeker_reward -= 0.02 
                
                # --- Штраф за стіни ---
                if s_sensors:
                    min_wall_dist = min(s_sensors)
                    if min_wall_dist < 0.05: # Якщо до стіни менше 5% дистанції променя
                        seeker_reward -= 0.05 
                        
                # --- Штраф за рух задом ---
                if last_seeker_move < -0.1:
                    seeker_reward -= 0.03
                
                ue_reward = msg.get("reward", 0.0)
                if abs(ue_reward) > 1.0: 
                    seeker_reward += ue_reward
                
                hider_reward = -seeker_reward
                # ====================================================

                if is_done:
                    last_dist = None
                    last_seeker_move = 0.0 # Скидаємо рух для нового раунду
                    train_step(hider_model, hider_optimizer, hider_memory, "Hider")
                    train_step(seeker_model, seeker_optimizer, seeker_memory, "Seeker")
                    
                    rounds_played += 1
                    if rounds_played % 50 == 0:
                        # --- ЗМІНЕНО: Зберігаємо за абсолютними шляхами ---
                        torch.save(hider_model.state_dict(), HIDER_MODEL_PATH)
                        torch.save(seeker_model.state_dict(), SEEKER_MODEL_PATH)
                        logging.info(f"💾 ПРОГРЕС ЗБЕРЕЖЕНО! (Пройдено раундів: {rounds_played})")
                    
                    out_msg = {"ok": True, "hider_move": 0.0, "hider_turn": 0.0, "seeker_move": 0.0, "seeker_turn": 0.0}
                    out = json.dumps(out_msg).encode("utf-8")
                    conn.sendall(struct.pack("!I", len(out)) + out)
                    continue

                h_vel = obs.get("hider_vel", [0.0, 0.0])
                s_vel = obs.get("seeker_vel", [0.0, 0.0])

                # Hider бачить: своя позиція, позиція seeker'а, своя швидкість, дистанція, свій радар, свої сенсори
                hider_state = [
                    obs["hider"][0] / MAX_POS, obs["hider"][1] / MAX_POS,
                    obs["seeker"][0] / MAX_POS, obs["seeker"][1] / MAX_POS,
                    h_vel[0] / MAX_VEL, h_vel[1] / MAX_VEL,
                    dist, hider_radar
                ] + h_sensors

                # Seeker бачить: своя позиція, позиція hider'а, своя швидкість, дистанція, свій радар, свої сенсори
                seeker_state = [
                    obs["seeker"][0] / MAX_POS, obs["seeker"][1] / MAX_POS,
                    obs["hider"][0] / MAX_POS, obs["hider"][1] / MAX_POS,
                    s_vel[0] / MAX_VEL, s_vel[1] / MAX_VEL,
                    dist, seeker_radar
                ] + s_sensors

                hider_state_tensor  = torch.FloatTensor(hider_state).unsqueeze(0).to(device)
                seeker_state_tensor = torch.FloatTensor(seeker_state).unsqueeze(0).to(device)

                # Inference без збереження графів — torch.no_grad() для самплювання
                with torch.no_grad():
                    h_mean, h_std = hider_model(hider_state_tensor)
                    h_action = Normal(h_mean, h_std).rsample()
                    h_action_clipped = torch.clamp(h_action, -1.0, 1.0).squeeze().tolist()

                    s_mean, s_std = seeker_model(seeker_state_tensor)
                    s_action = Normal(s_mean, s_std).rsample()
                    s_action_clipped = torch.clamp(s_action, -1.0, 1.0).squeeze().tolist()

                # Зберігаємо стани та дії на CPU без графа — graph-free, без витоку VRAM
                hider_memory.states.append(hider_state_tensor.detach().squeeze(0).cpu())
                hider_memory.actions.append(h_action.detach().squeeze(0).cpu())
                hider_memory.rewards.append(hider_reward)

                seeker_memory.states.append(seeker_state_tensor.detach().squeeze(0).cpu())
                seeker_memory.actions.append(s_action.detach().squeeze(0).cpu())
                seeker_memory.rewards.append(seeker_reward)

                # Оновлюємо змінну для перевірки на наступному кадрі
                last_seeker_move = s_action_clipped[0]

                # Страховка: якщо епізод дуже довгий — тренуємось і продовжуємо
                if len(hider_memory.rewards) >= MAX_STEPS_PER_EPISODE:
                    logging.info(f"[ЛІМІТ КРОКІВ] Досягнуто {MAX_STEPS_PER_EPISODE} кроків — проміжне навчання.")
                    train_step(hider_model,  hider_optimizer,  hider_memory,  "Hider")
                    train_step(seeker_model, seeker_optimizer, seeker_memory, "Seeker")
                
                out_msg = {
                    "ok": True, 
                    "hider_move": h_action_clipped[0],
                    "hider_turn": h_action_clipped[1],
                    "seeker_move": s_action_clipped[0],
                    "seeker_turn": s_action_clipped[1]
                }
                
                logging.info(f"X={obs['seeker'][0]:.1f}, Y={obs['seeker'][1]:.1f} | Дії Мисливця: Рух={s_action_clipped[0]:.2f}, Поворот={s_action_clipped[1]:.2f} | Радар={seeker_radar:.2f} | Нагорода={seeker_reward:.4f}")
                
                out = json.dumps(out_msg).encode("utf-8")
                conn.sendall(struct.pack("!I", len(out)) + out)
                
            except Exception as e:
                logging.error(f"Processing error: {e}")
                out_msg = {"ok": False, "err": str(e), "hider_move": 0.0, "hider_turn": 0.0, "seeker_move": 0.0, "seeker_turn": 0.0}
                out = json.dumps(out_msg).encode("utf-8")
                conn.sendall(struct.pack("!I", len(out)) + out)

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(16)
        logging.info(f"Continuous MARL Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            try:
                handle_client(conn, addr)
            except Exception:
                traceback.print_exc()

if __name__ == "__main__":
    main()