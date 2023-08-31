from evaluation import evaluate
from model import EmbeddingNet, TripletLoss, TripletDataset

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
# Create a directory to save your results
RESULTS_DIRECTORY = 'triplet-results-visual-query'
# Number of epochs and margin for triplet loss
EPOCHS = 20001
MARGIN = 0.5

# Set device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_triplet_loss(epochs=EPOCHS, batch_size=1):
    print("STARTING TRAINING...")

    #Create a directory to save your results
    if os.path.exists(RESULTS_DIRECTORY): 
        raise Exception(f"Directory {RESULTS_DIRECTORY} already exists, please delete it before running this script again.")

    print(f"Directory {RESULTS_DIRECTORY} does not exist, creating...")
    os.makedirs(RESULTS_DIRECTORY)

    # Ask the user for input
    # user_input = input("Please enter any identification information about this training: ")
    # Open the file in write mode ('w')
    with open(f"{RESULTS_DIRECTORY}/information.txt", "w") as file:
        # Write the user's input to the file
        # file.write(user_input)
        # file.write("\n")
        file.write(f"Query is Visual. Tactile+Audio as Retrieval. Classifiaction with vision. Training with margin {MARGIN} and {epochs} epochs, and batch size {batch_size}.")


    # Load your embeddings
    query_embeddings = np.load("/scratch/users/k21171248/c-entropy-results-visual-query/visual_embeddings_train.npy", allow_pickle=True).item()
    fused_embeddings = np.load("/scratch/users/k21171248/c-entropy-results-visual-query/tactile_audio_fused_train.npy", allow_pickle=True).item()
    query_embeddings_test = np.load("/scratch/users/k21171248/c-entropy-results-visual-query/visual_embeddings_test.npy", allow_pickle=True).item()  
    fused_embeddings_test = np.load("/scratch/users/k21171248/c-entropy-results-visual-query/tactile_audio_fused_test.npy", allow_pickle=True).item()  

    # Instantiate your dataset and dataloader
    triplet_dataset = TripletDataset(query_embeddings, fused_embeddings)
    triplet_dataloader = DataLoader(triplet_dataset, batch_size, shuffle=True)

    # Initialize loss function
    triplet_loss = TripletLoss(margin=MARGIN)

    model = EmbeddingNet(embedding_dim=200).to(device)
    # Initialize your optimizer, suppose you have a model named "model"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create a directory to save your results
    results_map = {
        'fused2query': [],
        'query2fused': []
    }
    triplet_loss_save = {
        'triplet_loss': []
    }

    best_map_pairs = {
        'MAP_pairs': []
    }

    # Initialize max MAP values to get best MAP results during training
    max_query2fused = 0.0
    max_fused2query = 0.0

    # Start training loop
    for epoch in range(epochs):
        total_loss = 0

        for i, (anchor, positive, negative, label) in enumerate(triplet_dataloader):
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Pass data through the model
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            # Compute the loss
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Add loss
            total_loss += loss.item()

        avg_loss = total_loss / len(triplet_dataloader)
        triplet_loss_save['triplet_loss'].append(avg_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCHS, avg_loss))

        if epoch % 100 == 0:
            new_query_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in query_embeddings.items()}
            new_fused_embeddings = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in fused_embeddings.items()}
            
            with torch.no_grad():
                new_query_embeddings_test = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in query_embeddings_test.items()}
                new_fused_embeddings_test = {k: model(torch.tensor(v, device=device)).detach().cpu().numpy() for k, v in fused_embeddings_test.items()}

            
            # Evaluate embeddings
            MAP_fused2query, MAP_query2fused = evaluate(new_query_embeddings_test, new_fused_embeddings_test, new_query_embeddings, new_fused_embeddings)
            
            if MAP_fused2query > max_fused2query:
                max_fused2query = MAP_fused2query
                best_map_pairs['MAP_pairs'].append((epoch, MAP_fused2query, MAP_query2fused))
                np.save('{}/trained_query_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_query_embeddings)
                np.save('{}/trained_fused_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_fused_embeddings)
                torch.save(model.state_dict(), f"{RESULTS_DIRECTORY}/model_best_fused2query.pth")
                
            if MAP_query2fused > max_query2fused:
                max_query2fused = MAP_query2fused
                best_map_pairs['MAP_pairs'].append((epoch, MAP_fused2query, MAP_query2fused))
                np.save('{}/trained_query_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_query_embeddings)
                np.save('{}/trained_fused_embeddings_{}.npy'.format(RESULTS_DIRECTORY, epoch), new_fused_embeddings)
                torch.save(model.state_dict(), f"{RESULTS_DIRECTORY}/model_best_query2fused.pth")


            # Add the results to the map
            results_map['fused2query'].append(MAP_fused2query)
            results_map['query2fused'].append(MAP_query2fused)

    # Save the map results as a JSON file
    with open('{}/map_results_{}.json'.format(RESULTS_DIRECTORY, epoch), 'w') as f:
        json.dump(results_map, f)
    with open('{}/triplet_loss.json'.format(RESULTS_DIRECTORY), 'w') as f:
        json.dump(triplet_loss_save, f)
    with open('{}/best_map_pairs.json'.format(RESULTS_DIRECTORY), 'w') as f:
        json.dump(best_map_pairs, f)

    # Plot the results
    plt.figure(figsize=(12,6))
    plt.plot(range(len(results_map['fused2query'])), results_map['fused2query'], label='Fused to Query')
    plt.plot(range(len(results_map['query2fused'])), results_map['query2fused'], label='Query to Fused')
    plt.xlabel('Triplet Loss Training Epoch')
    plt.ylabel('MAP')
    plt.legend()
    plt.title('MAP Results - Triplet Loss Training')
    plt.savefig('{}/map_plot_{}.png'.format(RESULTS_DIRECTORY, epoch))
    plt.close()

    #Print best results and save them to an information file
    print('MAP Fused to Query: {}'.format(max_fused2query))
    print('MAP Query to Fused: {}'.format(max_query2fused))

    with open(f"{RESULTS_DIRECTORY}/information.txt", "a") as file:
        # Write the user's input to the file
        file.write(f"\nMAP Fused to Query: {max_fused2query}")
        file.write(f"\nMAP Query to Fused: {max_query2fused}")

    # Plot the triplet loss
    plt.figure(figsize=(12,6))
    plt.plot(range(len(triplet_loss_save['triplet_loss'])), triplet_loss_save['triplet_loss'], label='Triplet Loss')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Triplet Loss', fontsize=18)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Triplet Loss Training', fontsize=18)
    plt.savefig(f'{RESULTS_DIRECTORY}/triplet_loss_plot.png')
    plt.close()


if __name__ == '__main__':
    train_with_triplet_loss()


