import random

import models
import utils
from options import CustomParser
from utils import *

config = CustomParser().parse({})

# set all random seeds for reproducibility
torch.manual_seed(config["random_seed"])
torch.cuda.manual_seed(config["random_seed"])
random.seed(config["random_seed"])

# set device
if config["device"] == "cuda" and torch.cuda.is_available():
    config["device"] = torch.device("cuda:0")
else:
    config["device"] = torch.device("cpu")

wandb.init(project="AE_project", entity="arafa42", name=config["name"] + "_train")

# Let's first prepare the MNIST dataset,
# run the test_dataset.py file to view some examples and see the dimensions of your tensor.
dataset = Dataset(config)

# TODO: define and train the model. Use the function from utils.py
print(config)
model = models.VanillaAutoEncoder(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, eps=1e-08, weight_decay=0.01)
utils.train_autoencoder(model, config, dataset, optimizer)

# save the model
save(model, config)

# display some images with its reconstruction
log = {}

recon = test_autoencoder(model, dataset, config)
log.update(
    {"Image reconstruction": wandb.Image(recon)}
)

gen = generate_using_encoder(model, config)
log.update(
    {"Image generation": wandb.Image(dataset.denormalize(gen))}
)

wandb.log(log)
