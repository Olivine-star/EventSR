import torch
import os


def checkpoint_restore(model, path, name="ckpt", device='cuda'):
    model.cpu()
    f = os.path.join(path, name + '.pth')

    # 路径检查提示
    print(f"[INFO] Looking for checkpoint at: {f}")

    if not os.path.exists(path):
        print(f"[WARNING] Path '{path}' does not exist. Creating it.")
        os.makedirs(path)

    if os.path.exists(f):
        print(f"✅ Loaded checkpoint from: {f}")
        model_CKPT = torch.load(f)
        model.load_state_dict(model_CKPT['state_dic'])

        if 'epoch' in model_CKPT:
            epoch = model_CKPT['epoch']
            print(f"[INFO] Loaded model from epoch {epoch}")
        else:
            epoch = -1
            print("[WARNING] Epoch not found in checkpoint, defaulting to -1")
    else:
        print(f"\n[ERROR] Checkpoint file not found: {f}")
        raise FileNotFoundError(f"Checkpoint not found at {f}")

    model.to(device)
    return model, epoch



def checkpoint_save(model, path, epoch, name="ckpt", device='cuda'):
    if not os.path.exists(path):
        os.makedirs(path)
    if name == "ckptBest":
        f = os.path.join(path, name + '.pth')
    else:
        f = os.path.join(path, name + str(epoch) + '.pth')
    model.cpu()
    torch.save({'state_dic': model.state_dict(), "epoch": epoch}, f)
    model.to(device)