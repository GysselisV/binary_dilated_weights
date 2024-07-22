import torch
from .base_model import BaseModel
from . import networks
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class pix2pixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #Máscaras
        self.mask = input['mask'].to(self.device)
        self.brain_mask = input['brain_mask'].to(self.device)


        def sample_weights_torch(mask, class_weights):
            indices = mask.cpu().numpy().astype(np.int64)
            return torch.from_numpy(np.take(class_weights, indices)).to(mask.device)
        
        def dilated_sample_weights_torch(mask, inf_val=0.1, sup_val=0.8, steps=10):
            # mask: forma correcta [batch_size, channels, height, width]
            # Si mask tiene más dimensiones de las esperadas, las eliminamos
            while mask.dim() > 4:
                mask = mask.squeeze(0)
            weights = torch.ones_like(mask) * inf_val
            weights[mask == 1] = sup_val
            kernel = torch.ones((1, 1, 3, 3), device=mask.device)
            for i in range(1, steps):
                dilated_mask = (F.conv2d(mask.float(), kernel, padding=1) > 0).float()
                new_weights = torch.where(dilated_mask - mask > 0, (sup_val - inf_val) / steps * i + inf_val, weights)
                weights = new_weights
                mask = dilated_mask
            return weights

        class_weights = [1.0, 3.0, 10.0]  #Fondo, cerebro, lesión
    
        # Calcular pesos combinados
        combined_mask = self.mask + self.brain_mask
        weights = sample_weights_torch(combined_mask, class_weights) 
        
        # Dilatados 
        dilated_weights = dilated_sample_weights_torch(self.mask, inf_val=class_weights[1], sup_val=class_weights[2])
        dilated_weights = dilated_weights * self.brain_mask
        dilated_weights[self.brain_mask == 0] = class_weights[0]

        self.weights = weights
        self.dilated_weights = dilated_weights

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        #Visualizar
        # self.visualize_masks_and_weights()


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        

        l1_loss_per_pixel = torch.abs(self.fake_B - self.real_B)  # Diferencia absoluta por píxel
        weighted_l1_loss = l1_loss_per_pixel * self.dilated_weights  # Aplicar pesos
        self.loss_G_L1 = weighted_l1_loss.mean() * self.opt.lambda_L1  # Media y escalado por lambda5


        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights


    def visualize_masks_and_weights(self):
        masks = [self.real_A.cpu().numpy(), self.real_B.cpu().numpy(), self.mask.cpu().numpy(), self.brain_mask.cpu().numpy(), self.weights.cpu().numpy(), self.dilated_weights.cpu().numpy()]
        titles = ['Input Image (A)', 'Target Image (B)', 'Lesion Mask', 'Brain Mask', 'Binary', 'Dilated Weights']
        os.makedirs('visualizations', exist_ok=True)
        
        for i, mask in enumerate(masks):
            max_val = mask.max()
            min_val = mask.min()
            print(f'{titles[i]} - Max: {max_val}, Min: {min_val}')
            
            plt.figure(figsize=(10, 10))
            plt.title(f'{titles[i]} (Max: {max_val}, Min: {min_val})')
            plt.imshow(mask[0, 0, :, :], cmap='gray')
            plt.colorbar()
            plt.savefig(f'visualizations/{titles[i].replace(" ", "_")}.png')
            plt.close()