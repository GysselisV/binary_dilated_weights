import time
from options.train_options import TrainOptions
from models import create_model
from data import create_dataset
import torch
opt = TrainOptions().parse()
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)

model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
total_iters = 0                # the total number of training iterations

for i, data in enumerate(dataset):
    print(type(data))
    print(data["A"].shape, data["A"].min(), data["A"].max(), data["A"].mean())
    print(data["B"].shape, data["B"].min(), data["B"].max(), data["B"].mean())
    model.set_input(data)         # unpack data from dataset and apply preprocessing
    model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
    print(type(data))
    print(data.keys())
    print(type(data["A"]))
    print(type(data["B"]))
    print(data["A"].shape, data["A"].min(), data["A"].max(), data["A"].mean())
    print(data["B"].shape, data["B"].min(), data["B"].max(), data["B"].mean())

    # Crear instancia de la perdida.
    loss = torch.nn.L1Loss(reduction="none")
    
    print(loss_result.shape, loss_result.min(), loss_result.max())
    # Cargar una mascara.
    masks = "home/data/gustavo_pupils/Gyss/trabajo_pregrado/combined_dataset/train_png/train_009/MASK/train_009_018.png"
    print(mask.shape)
    # A partir de la mascara, crear los class weights.
    class_weights = get_class_weights(mask)

    loss_result = loss(data["B"], data["B"])
    loss_result = loss_result * class_weights
    loss_result = loss_result.mean()
    
    break
