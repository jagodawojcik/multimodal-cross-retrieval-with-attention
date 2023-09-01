from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model import TactileNetwork, CrossSensoryNetwork
from load_data import get_loader
import json
import os

EPOCHS_PRETRAIN = 15
EPOCHS_C_ENTROPY = 50
BATCH_SIZE = 5

RESULTS_DIRECTORY = 'c-entropy-results-visual-query'

def train_with_cross_entropy(epochs_pre = EPOCHS_PRETRAIN, epochs_cross_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE):

    #Create a directory to save your results
    if os.path.exists(RESULTS_DIRECTORY): 
        raise Exception(f"Directory {RESULTS_DIRECTORY} already exists, please delete it before running this script again.")

    print(f"Directory {RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(RESULTS_DIRECTORY)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize your Tactile Network
    tactile_network = TactileNetwork().to(device)

    # Initialize your optimizer and loss function for the pretraining
    pretrain_optimizer = torch.optim.Adam(tactile_network.parameters(), lr=0.001)
    pretrain_criterion = nn.CrossEntropyLoss()

    # Get the dataloaders and parameters
    dataloader, input_data_par = get_loader(batch_size)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    # # Pretraining loop
    # tactile_embeddings_pretrain = defaultdict(list)

    # Initialize list to store losses
    train_losses = []
    test_losses = []
    test_losses_audio = []

    for epoch in range(epochs_pre):
        tactile_network.train()  # set network to training mode
        total_loss = 0

        for i, (_, tactile_input, _, targets) in enumerate(train_loader):
            tactile_input, targets = tactile_input.to(device), targets.to(device)

            pretrain_optimizer.zero_grad()

            # Get outputs and embeddings
            tactile_output = tactile_network.tactile_branch(tactile_input)
            outputs = tactile_network.fc(tactile_output)

            # Compute the loss
            loss = pretrain_criterion(outputs, targets)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            pretrain_optimizer.step()

            # Save embeddings for each batch
            for j in range(tactile_output.shape[0]):
                label = targets[j].item()
                # tactile_embeddings_pretrain[label].append(tactile_output[j].detach().cpu().numpy())
            
        # End of epoch
        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)
        print(f'Pretraining Epoch {epoch}, Train Loss: {train_loss}')

        # Evaluation loop on test set
        tactile_network.eval()  # set network to evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for i, (_, tactile_input, _, targets) in enumerate(test_loader):
                tactile_input, targets = tactile_input.to(device), targets.to(device)
                tactile_output = tactile_network.tactile_branch(tactile_input)
                outputs = tactile_network.fc(tactile_output)
                test_loss = pretrain_criterion(outputs, targets)
                total_test_loss += test_loss.item()

        test_loss = total_test_loss/len(test_loader)
        test_losses.append(test_loss)
        print(f'Pretraining Epoch {epoch}, Test Loss: {test_loss}')

    # Save the model
    torch.save(tactile_network.state_dict(), f"{RESULTS_DIRECTORY}/model_query2fused.pth")
    
    # Plot train and test loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Loss Metrics', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f"{RESULTS_DIRECTORY}/pretrain_loss_plot.png")
    plt.show()
    # # Save the embeddings after pretraining
    # print("Pretraining completed. Saving pretrain tactile embeddings...")
    # np.save('tactile_embeddings_pretrain.npy', dict(tactile_embeddings_pretrain))
        
    network = CrossSensoryNetwork().to(device)

    # Load the pretrained weights into the tactile branch
    network.tactile_branch.load_state_dict(tactile_network.tactile_branch.state_dict())

    # Initialize your optimizer and loss function for the main training
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store losses
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(epochs_cross_entropy):
        network.train()  # set network to training mode
        total_train_loss = 0

        # Initialize embeddings storage for each epoch
        audio_embeddings_train = defaultdict(list)
        tactile_embeddings_train = defaultdict(list)
        visual_embeddings_train = defaultdict(list)
        visual_tactile_fused_train = defaultdict(list)
        # Training phase
        for i, (audio_input, tactile_input, visual_input, targets) in enumerate(train_loader):
            audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            audio_output, tactile_output, visual_output, attention_out, joint_embeddings = network(audio_input, tactile_input, visual_input)

            # Compute the loss
            loss = criterion(tactile_output, targets)
            total_train_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save embeddings for each batch
            for j in range(audio_output.shape[0]):
                label = targets[j].item()
                audio_embeddings_train[label].append(audio_output[j].detach().cpu().numpy())
                tactile_embeddings_train[label].append(tactile_output[j].detach().cpu().numpy())
                visual_embeddings_train[label].append(visual_output[j].detach().cpu().numpy())
                visual_tactile_fused_train[label].append(attention_out[j].detach().cpu().numpy())
        
        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)  # Append training loss for current epoch

        # Evaluation phase on test set
        network.eval()  # set network to evaluation mode
        audio_embeddings_test = defaultdict(list)
        tactile_embeddings_test = defaultdict(list)
        visual_embeddings_test = defaultdict(list)
        visual_tactile_fused_test = defaultdict(list)

        total_test_loss = 0
        total_test_loss_audio = 0
        with torch.no_grad():
            for i, (audio_input, tactile_input, visual_input, targets) in enumerate(test_loader):
                audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

                # Get outputs and embeddings
                audio_output, tactile_output, visual_output, attention_out, joint_embeddings = network(audio_input, tactile_input, visual_input)

                # Compute the loss
                loss = criterion(joint_embeddings, targets)
                total_test_loss += loss.item()
                # Compute the loss but on audio
                audio_loss = criterion(tactile_output, targets)  
                total_test_loss_audio += audio_loss.item() 

                # Save test embeddings for each batch
                for j in range(audio_output.shape[0]):
                    label = targets[j].item()
                    audio_embeddings_test[label].append(audio_output[j].detach().cpu().numpy())
                    tactile_embeddings_test[label].append(tactile_output[j].detach().cpu().numpy())
                    visual_embeddings_test[label].append(visual_output[j].detach().cpu().numpy())
                    visual_tactile_fused_test[label].append(attention_out[j].detach().cpu().numpy())

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)  # Append test loss for current epoch
        print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {test_loss}')

        audio_test_loss = total_test_loss_audio / len(test_loader) 
        test_losses_audio.append(audio_test_loss)  # <- Append audio-specific test loss for current epoch

        print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {test_loss}, tactile Test Loss: {audio_test_loss}')  # <- Update the print statement


    # Save the embeddings after all epochs
    print("Training completed. Saving embeddings...")
    np.save(f"{RESULTS_DIRECTORY}/audio_embeddings_train.npy", dict(audio_embeddings_train))
    np.save(f"{RESULTS_DIRECTORY}/tactile_embeddings_train.npy", dict(tactile_embeddings_train))
    np.save(f"{RESULTS_DIRECTORY}/visual_embeddings_train.npy", dict(visual_embeddings_train))
    np.save(f"{RESULTS_DIRECTORY}/tactile_audio_fused_train.npy", dict(visual_tactile_fused_train))
    np.save(f"{RESULTS_DIRECTORY}/audio_embeddings_test.npy", dict(audio_embeddings_test))
    np.save(f"{RESULTS_DIRECTORY}/tactile_embeddings_test.npy", dict(tactile_embeddings_test))
    np.save(f"{RESULTS_DIRECTORY}/visual_embeddings_test.npy", dict(visual_embeddings_test))
    np.save(f"{RESULTS_DIRECTORY}/tactile_audio_fused_test.npy", dict(visual_tactile_fused_test))
    # Save the trained model
    torch.save(network.state_dict(), f"{RESULTS_DIRECTORY}/audio-visual-tactile-model.pth")
    
    # Save train and test losses to a JSON file
    loss_dict = {'train_losses': train_losses, 'test_losses': test_losses, 'test_audio_losses': test_losses_audio}
    with open(f"{RESULTS_DIRECTORY}/c_entropy_train_test_losses.json", 'w') as f:
        json.dump(loss_dict, f)  # <- Save losses as a JSON file

    # After training, plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')

    # Increase title font size
    plt.title('Train and Test Loss over time', fontsize=18)

    # Increase x and y axis label font size
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)

    # Increase tick font size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Increase legend font size
    plt.legend(fontsize=16)

    plt.show()

    # Save the figure
    plt.savefig(f"{RESULTS_DIRECTORY}/train_test_loss_plot.png")

    # Display the plot
    plt.show()

