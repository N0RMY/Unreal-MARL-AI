import socket
import json

HOST = "127.0.0.1"
PORT = 5555

def decide_action(agent, obs):
    # Поки що "заглушка":
    # seeker — біжить "у бік" (умовно) ворога
    # hider — тікає в протилежний бік
    # Приклад: obs = [dir_x, dir_y, dist_norm, speed_norm]
    dx = obs[0]
    dy = obs[1]

    if agent == "seeker":
        ax, ay = dx, dy
    else:  # hider
        ax, ay = -dx, -dy

    # clamp
    ax = max(-1.0, min(1.0, float(ax)))
    ay = max(-1.0, min(1.0, float(ay)))
    return [ax, ay]

def main():
    print(f"Starting RL server on {HOST}:{PORT}")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)

    conn, addr = s.accept()
    print("Client connected:", addr)

    buf = b""
    while True:
        data = conn.recv(4096)
        if not data:
            print("Client disconnected.")
            break
        buf += data

        # Протокол: кожне повідомлення закінчується '\n'
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue

            req = json.loads(line.decode("utf-8"))
            agent = req.get("agent", "hider")
            obs = req.get("obs", [0, 0, 0, 0])

            action = decide_action(agent, obs)
            resp = {"action": action}

            conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

if __name__ == "__main__":
    main()
