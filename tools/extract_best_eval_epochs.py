import re
from pathlib import Path

runs = {
    "enclosing-only 100e": "logs/Train_quad_drone_Val_quad_drone/0619_1252/log.txt",
    "prop-proto 100e": "logs/Train_quad_drone_Val_quad_drone/0619_2009/log.txt",
}

# 这两个 run 都是从 epoch 23 resume，val_freq=5，
# 正常 eval epoch 应该是 25, 30, ..., 100
def infer_epoch(idx):
    return 25 + idx * 5

for name, path in runs.items():
    print("\n====", name, "====")
    path = Path(path)
    if not path.exists():
        print("Missing:", path)
        continue

    rows = []
    cur_epoch = None
    acc25 = None
    acc50 = None
    miou = None

    with path.open("r", errors="ignore") as f:
        for line in f:
            m = re.search(r"Eval epoch\s+(\d+)", line)
            if m:
                cur_epoch = int(m.group(1))

            m = re.search(r"Eval/acc@0\.25:\s*([0-9.]+)", line)
            if m:
                acc25 = float(m.group(1))

            m = re.search(r"Eval/acc@0\.5:\s*([0-9.]+)", line)
            if m:
                acc50 = float(m.group(1))

            # 避免匹配到别的东西，只取形如 “mIoU: 0.xxxx”
            m = re.search(r"\bmIoU:\s*([0-9.]+)", line)
            if m:
                miou = float(m.group(1))

            if acc25 is not None and acc50 is not None and miou is not None:
                epoch = cur_epoch if cur_epoch is not None else infer_epoch(len(rows))
                rows.append((epoch, acc25, acc50, miou))
                acc25 = acc50 = miou = None

    if not rows:
        print("No eval rows found. Try checking log manually:")
        print(f"grep -n \"Eval/acc@0.25\\|Eval/acc@0.5\\|mIoU\" {path}")
        continue

    for epoch, a25, a50, miou in rows:
        print(f"epoch {epoch:3d}: acc25={a25:.4f}, acc50={a50:.4f}, mIoU={miou:.4f}")

    best25 = max(rows, key=lambda x: x[1])
    best50 = max(rows, key=lambda x: x[2])
    bestmiou = max(rows, key=lambda x: x[3])

    print("\nBEST acc@0.25:", f"epoch {best25[0]}, acc25={best25[1]:.4f}, acc50={best25[2]:.4f}, mIoU={best25[3]:.4f}")
    print("BEST acc@0.5 :", f"epoch {best50[0]}, acc25={best50[1]:.4f}, acc50={best50[2]:.4f}, mIoU={best50[3]:.4f}")
    print("BEST mIoU    :", f"epoch {bestmiou[0]}, acc25={bestmiou[1]:.4f}, acc50={bestmiou[2]:.4f}, mIoU={bestmiou[3]:.4f}")
