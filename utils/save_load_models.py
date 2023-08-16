import torch
import os
import json

f = open("configs.json")
configs = json.load(f)

root_gen_XY_path = configs["path"]["root_gen_XY_path"]
root_dis_X_path = configs["path"]["root_dis_X_path"]
root_gen_YX_path = configs["path"]["root_gen_YX_path"]
root_dis_Y_path = configs["path"]["root_dis_Y_path"]

if (
    int(os.listdir(root_gen_XY_path)[-1])
    == int(os.listdir(root_dis_X_path)[-1])
    == int(os.listdir(root_gen_YX_path)[-1])
    == int(os.listdir(root_dis_Y_path)[-1])
):
    global current_model_num
    current_model_num = int(os.listdir(root_gen_XY_path)[-1])
else:
    raise FileExistsError("Ambiguity in the current model number.")

root_gen_path_A = root_gen_XY_path + "/" + str(current_model_num + 1)
root_dis_path_A = root_dis_X_path + "/" + str(current_model_num + 1)
root_gen_path_B = root_gen_YX_path + "/" + str(current_model_num + 1)
root_dis_path_B = root_dis_Y_path + "/" + str(current_model_num + 1)

os.mkdir(root_gen_path_A)
os.mkdir(root_dis_path_A)
os.mkdir(root_gen_path_B)
os.mkdir(root_dis_path_B)


def save_model(
    gen_model_A, dis_model_A, gen_model_B, dis_model_B, epoch, path: tuple = None
) -> None:
    if path is None:
        gen_path_A = root_gen_path_A + "/" + str(epoch) + ".pth"
        dis_path_A = root_dis_path_A + "/" + str(epoch) + ".pth"
        gen_path_B = root_gen_path_B + "/" + str(epoch) + ".pth"
        dis_path_B = root_dis_path_B + "/" + str(epoch) + ".pth"
        torch.save(gen_model_B.state_dict(), gen_path_B)
        torch.save(dis_model_B.state_dict(), dis_path_B)
        torch.save(gen_model_A.state_dict(), gen_path_A)
        torch.save(dis_model_A.state_dict(), dis_path_A)
    else:
        torch.save(gen_model_A.state_dict(), path[0])
        torch.save(dis_model_A.state_dict(), path[1])
        torch.save(gen_model_B.state_dict(), path[2])
        torch.save(dis_model_B.state_dict(), path[3])


def load_model(path: tuple) -> tuple:
    gen_model_A = torch.load(path[0])
    dis_model_A = torch.load(path[1])
    gen_model_B = torch.load(path[2])
    dis_model_B = torch.load(path[3])
    return gen_model_A, dis_model_A, gen_model_B, dis_model_B
