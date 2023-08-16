import json
import torch
import argparse
import cv2 as cv
from gen_critic import Generator
from utils.dataloader import load_validation_images, get_dataset_info

parser = argparse.ArgumentParser(description="Test the trained model.")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Specify the pth path of trained model.",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

f = open("configs.json")
configs = json.load(f)

height = configs["img_specifications"]["height"]
width = configs["img_specifications"]["width"]
outputs_path = configs["path"]["model_outputs"]
number_of_validation_images = get_dataset_info()["number_of_validation_images"]

print(f"number of validation images = {number_of_validation_images}")

generator_XY = Generator().to(device)


def test(model_path):
    generator_XY.load_state_dict(torch.load(model_path))
    for i in range(number_of_validation_images):
        X, Y = load_validation_images(i)
        Y_cap = generator_XY(X)
        cv.imwrite(
            outputs_path + "/testing_output/Y_" + str(i) + ".jpg",
            (Y.reshape(height, width, 3).detach().cpu().numpy() + 1) * (127.5),
        )

        cv.imwrite(
            outputs_path + "/testing_output/Y_cap_" + str(i) + ".jpg",
            (Y_cap.reshape(height, width, 3).detach().cpu().numpy() + 1) * (127.5),
        )
        print(f"outputed {i+1}/{number_of_validation_images}")


if __name__ == "__main__":
    test(args.model_path)
