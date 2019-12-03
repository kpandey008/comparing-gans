### WGAN related loss functions
def w_disc_loss(real_samples, noise_samples, disc_model, gen_model):
    real_output = disc_model(real_samples)
    fake_samples = gen_model(noise_samples)
    fake_output = disc_model(fake_samples)
    return -torch.mean(real_output - fake_output)


def w_disc_loss_gp(real_samples, noise_samples, epsilon, reg_param, disc_model, gen_model):
    real_output = disc_model(real_samples)
    fake_samples = gen_model(noise_samples)
    fake_output = disc_model(fake_samples)
    critic_loss = -torch.mean(real_output - fake_output)
    
    # Compute the gradient penalty
    aux_samples = epsilon * real_samples + (1 - epsilon) * fake_samples
    aux_outputs = disc_model(aux_samples)
    gradients = grad(outputs=aux_outputs,
                     inputs=aux_samples,
                     grad_outputs=torch.ones(aux_outputs.shape, device=device),
                     create_graph=True,
                     retain_graph=True)[0]
    gradient_penalty = reg_param * (torch.pow(torch.norm(gradients, 2), 2) - 1)
    return critic_loss + gradient_penalty


def w_gen_loss(noise_samples, disc_model, gen_model):
    fake_samples = gen_model(noise_samples)
    fake_output = disc_model(fake_samples)
    return -torch.mean(fake_output)


## LSGAN related loss functions
def ls_disc_loss(real_samples, noise_samples, disc_model, gen_model):
    real_output = disc_model(real_samples)
    real_loss = torch.mean(torch.pow(real_output - 1, 2))

    fake_samples = gen_model(noise_samples)
    fake_output = disc_model(fake_samples)
    fake_loss = torch.mean(torch.pow(fake_output, 2))
    return real_loss + fake_loss


def ls_gen_loss(noise_samples, disc_model, gen_model):
    fake_samples = gen_model(noise_samples)
    fake_output = disc_model(fake_samples)
    return torch.mean(torch.pow(fake_output - 1, 2))


## Reconstruction Loss as proposed in the paper `Improved Training for GAN's`
def reconstruction_loss(data_batch, noise_samples, disc_model, gen_model):
    real_activation = disc_model.selective_forward('conv_2', data_batch)
    fake_samples = gen_model(noise_samples)
    fake_activation = disc_model.selective_forward('conv_2', fake_samples)
    statistics_loss = nn.MSELoss()
    return statistics_loss(fake_activation, real_activation)