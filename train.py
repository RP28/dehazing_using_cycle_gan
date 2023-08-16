import os
import json
import torch
import logging
import cv2 as cv
import torch.nn as nn
import torch.optim as optim
from gen_critic import Generator, Discriminator
from utils.save_load_models import save_model, load_model
from utils.dataloader import load_training_images, get_dataset_info

f = open("configs.json")
configs = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

g_lr = configs["training_params"]["g_lr"]
d_x_lr = configs["training_params"]["d_x_lr"]
d_y_lr = configs["training_params"]["d_y_lr"]
beta1 = configs["training_params"]["beta1"]
beta2 = configs["training_params"]["beta2"]
height = configs["img_specifications"]["height"]
width = configs["img_specifications"]["width"]
log_file_root_path = configs["path"]["logger_file"]
outputs_path = configs["path"]["model_outputs"]
batch_size = configs["training_params"]["batch_size"]
num_epochs = configs["training_params"]["num_epochs"]
num_of_training_images = get_dataset_info()["number_of_training_images"]

logging.basicConfig(
    filename=os.path.join(
        log_file_root_path,
        str(int(os.listdir(log_file_root_path)[-1][:-4]) + 1) + ".log",
    ),
    format="%(asctime)s %(message)s",
    filemode="w",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

print(f"Configurations:\n{configs}\n")
logger.info(f"Configurations:\n{configs}\n")

generator_XY = Generator().to(device)
generator_YX = Generator().to(device)
discriminator_X = Discriminator().to(device)
discriminator_Y = Discriminator().to(device)

adversarial_loss = nn.BCELoss()
cycle_consistency_loss = nn.L1Loss()

optimizer_G = optim.Adam(
    list(generator_XY.parameters()) + list(generator_YX.parameters()),
    lr=g_lr,
    betas=(beta1, beta2),
)
optimizer_D_X = optim.Adam(
    discriminator_X.parameters(), lr=d_x_lr, betas=(beta1, beta2)
)
optimizer_D_Y = optim.Adam(
    discriminator_Y.parameters(), lr=d_y_lr, betas=(beta1, beta2)
)

print(f"Number of training images = {num_of_training_images}\n")
logger.info(f"Number of training images = {num_of_training_images}\n")

for epoch in range(num_epochs):
    for i in range(0, num_of_training_images, batch_size):
        real_X, real_Y = load_training_images(i, batch_size)
        real_X, real_Y = real_X.to(device), real_Y.to(device)

        current_batch_dim = real_X.shape[0]

        real_labels = torch.ones((current_batch_dim, 1))
        fake_labels = torch.zeros((current_batch_dim, 1))

        optimizer_G.zero_grad()

        fake_Y = generator_XY(real_X)
        cycle_X = generator_YX(fake_Y)

        fake_X = generator_YX(real_Y)
        cycle_Y = generator_XY(fake_X)

        loss_G = (
            adversarial_loss(discriminator_Y(fake_Y), real_labels)
            + adversarial_loss(discriminator_X(fake_X), real_labels)
            + cycle_consistency_loss(cycle_X, real_X)
            + cycle_consistency_loss(cycle_Y, real_Y)
        )

        loss_G.backward()
        optimizer_G.step()

        optimizer_D_X.zero_grad()
        optimizer_D_Y.zero_grad()

        loss_D_X = adversarial_loss(
            discriminator_X(real_X), real_labels
        ) + adversarial_loss(discriminator_X(fake_X.detach()), fake_labels)

        loss_D_Y = adversarial_loss(
            discriminator_Y(real_Y), real_labels
        ) + adversarial_loss(discriminator_Y(fake_Y.detach()), fake_labels)

        loss_D_X.backward()
        loss_D_Y.backward()
        optimizer_D_X.step()
        optimizer_D_Y.step()

    print(
        f"Epoch [{epoch}/{num_epochs}] Loss_G: {loss_G.item()} Loss_D_X: {loss_D_X.item()} Loss_D_Y: {loss_D_Y.item()}"
    )
    logger.info(
        f"Epoch [{epoch}/{num_epochs}] Loss_G: {loss_G.item()} Loss_D_X: {loss_D_X.item()} Loss_D_Y: {loss_D_Y.item()}"
    )

    save_model(generator_XY, discriminator_X, generator_YX, discriminator_Y, epoch)

    cv.imwrite(
        outputs_path + "/fake_y.jpg",
        (fake_Y.reshape(height, width, 3).detach().cpu().numpy() + 1) * (127.5),
    )

    cv.imwrite(
        outputs_path + "/real_x.jpg",
        (real_X.reshape(height, width, 3).detach().cpu().numpy() + 1) * (127.5),
    )

    cv.imwrite(
        outputs_path + "/cycle_x.jpg",
        (cycle_X.reshape(height, width, 3).detach().cpu().numpy() + 1) * (127.5),
    )

    cv.imwrite(
        outputs_path + "/real_y.jpg",
        (real_Y.reshape(height, width, 3).detach().cpu().numpy() + 1) * (127.5),
    )


