import yaml
from torchvision import transforms


def _get_transforms():
    T = []
    T.append(transforms.ToTensor())
    return transforms.Compose(T)


def _load_config(config_path):
    with open(config_path, 'r') as reader:
        config = yaml.safe_load(reader)
    return config


def display_grid(batch, num_rows=8):
    """Displays a grid of samples
    """
    assert isinstance(batch, torch.Tensor)
    batch_size = batch.shape[0]
    num_cols = int(batch_size / num_rows)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 8))
    img_idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            img = batch[img_idx]
            ax[i, j].imshow(img)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            img_idx = img_idx + 1
    plt.show()


def _train_one_epoch(disc_model, gen_model, disc_loss, gen_loss, disc_optim, gen_optim, loader, k, epoch_idx, log_step=50):
    """Trains a GAN for one epoch
    """
    gen_loss_profile = []
    disc_loss_profile = []

    # Set the models in training mode
    disc_model.train()
    gen_model.train()

    # Train the models for 1 epoch (i.e over the entire dataset)
    for step_id, (data_batch, _) in enumerate(loader):
        
        # Move the data to the device
        data_batch = data_batch.to(device)
        for idx in range(k):
            noise_batch = torch.randn(batch_size, code_size).to(device)
            gen_output_batch = gen_model(noise_batch)
            fake_labels_batch = torch.zeros([batch_size, 1]).to(device)
            real_target_batch = torch.ones([batch_size, 1]).to(device)

            train_batch = torch.cat((gen_output_batch, data_batch), 0)
            train_labels = torch.cat((fake_labels_batch, real_target_batch), 0)

            # zero out any previous gradients
            disc_optim.zero_grad()

            # Train the discriminator
            preds = disc_model(train_batch)
            d_loss = disc_loss(preds, train_labels)
            d_loss.backward()
            disc_optim.step()

        # Train the Generator
        gen_optim.zero_grad()
        noise_batch = torch.randn(batch_size, code_size).to(device)
        gen_target_batch = torch.ones([batch_size, 1]).to(device)
        disc_preds = disc_model(gen_model(noise_batch))
        g_loss = gen_loss(disc_preds, gen_target_batch)
        g_loss.backward()
        gen_optim.step()

        if step_id % log_step == 0:
            clear_output()
            print(f'Training for epoch: {epoch_idx + 1}')
            print(f"G-Loss: {g_loss.item()} D-Loss: {d_loss.item()}")
            # display the dummy sample
            with torch.no_grad():
                sample_gen_output = gen_model(sample_noise.cuda())
                display_grid(sample_gen_output)

        # Create loss profiles for further visualization
        gen_loss_profile.append(g_loss.item())
        disc_loss_profile.append(d_loss.item())
    return gen_loss_profile, disc_loss_profile
