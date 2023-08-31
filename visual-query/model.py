import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from torch.nn import functional as F
import random
from torch.nn import MultiheadAttention

NUM_CLASSES = 20

"""Cross Entropy Network"""
##Visual an Tactile Branches
class ResnetBranch(nn.Module):
    """A network branch based on a pretrained ResNet."""
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(ResnetBranch, self).__init__()
        self.base_model = models.resnet50(pretrained=pre_trained)
    
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # remove the final fully connected layer
        self.fc = nn.Linear(num_features, output_dim)  # new fc layer for embeddings

    def forward(self, x):
        x = self.base_model(x)
        embeddings = self.fc(x)
        return embeddings
    
class AudioBranch(nn.Module):
    """A network branch based on 1D CNN for audio data."""
    def __init__(self, hidden_dim=2048, output_dim=200):
        super(AudioBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling layer to reduce the size
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Apply pooling after the convolutions
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# #Joint Branch
# class CrossSensoryNetwork(nn.Module):
#     def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
#         super(CrossSensoryNetwork, self).__init__()
#         self.tactile_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
#         self.audio_branch = AudioBranch(hidden_dim, output_dim)
#         self.visual_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
#         self.joint_fc = nn.Linear(output_dim * 2, NUM_CLASSES)  # for classification
#         self.attention = MultiheadAttention(embed_dim=output_dim, num_heads=8)

#     def forward(self, audio_input, tactile_input, visual_input):
#         tactile_output = self.tactile_branch(tactile_input)
#         audio_output = self.audio_branch(audio_input)
#         visual_output = self.visual_branch(visual_input)

#         # Use the tactile_output as the query to the attention mechanism
#         attn_out, _ = self.attention(visual_output.unsqueeze(0), audio_output.unsqueeze(0), tactile_output.unsqueeze(0))
#         attn_out = attn_out.squeeze(0)  # Remove the extra dimension

#         # Concatenation for classification
#         joint_representation = torch.cat([visual_output, attn_out], dim=1)
#         joint_classification_output = self.joint_fc(joint_representation)

#         return audio_output, tactile_output, visual_output, attn_out, joint_classification_output

class CrossSensoryNetwork(nn.Module):
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(CrossSensoryNetwork, self).__init__()
        self.tactile_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.audio_branch = AudioBranch(hidden_dim, output_dim)
        self.visual_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.joint_fc = nn.Linear(output_dim * 2, NUM_CLASSES)
        self.attention = MultiheadAttention(embed_dim=output_dim, num_heads=8)

    def forward(self, audio_input, tactile_input, visual_input):
        tactile_output = self.tactile_branch(tactile_input)
        audio_output = self.audio_branch(audio_input)
        visual_output = self.visual_branch(visual_input)

        # Bi-directional cross-attention
        attn_out1, _ = self.attention(audio_output.unsqueeze(0), visual_output.unsqueeze(0), visual_output.unsqueeze(0))
        attn_out2, _ = self.attention(visual_output.unsqueeze(0), audio_output.unsqueeze(0), audio_output.unsqueeze(0))

        # Remove the extra dimensions and combine
        attn_out1 = attn_out1.squeeze(0)
        attn_out2 = attn_out2.squeeze(0)
        attn_out_combined = (attn_out1 + attn_out2) / 2  # Average the two attention outputs

        # Concatenation for classification
        joint_representation = torch.cat([tactile_output, attn_out_combined], dim=1)
        joint_classification_output = self.joint_fc(joint_representation)

        return audio_output, tactile_output, visual_output, attn_out_combined, joint_classification_output
  
#Pretrain Branch for Tactile
class TactileNetwork(nn.Module):
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(TactileNetwork, self).__init__()
        self.tactile_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, NUM_CLASSES)  # final fc layer for classification

    def forward(self, tactile_input):
        tactile_output = self.tactile_branch(tactile_input)
        outputs = self.fc(tactile_output)
        return outputs
    
"""Triplet Loss Network"""
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 200)

    def forward(self, x):
        if isinstance(x, list):
            # Concatenate the list of embeddings along the batch dimension
            x = torch.cat(x, dim=0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2, dim=-1)  # normalize the embeddings to have norm=1
        return x

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletDataset(Dataset):
    def __init__(self, visual_embeddings, tactile_embeddings):
        assert visual_embeddings.keys() == tactile_embeddings.keys()

        self.labels = list(visual_embeddings.keys())
        self.visual_embeddings = visual_embeddings
        self.tactile_embeddings = tactile_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx] # get the label of the idx-th sample
        positive_source = random.choice(['visual', 'tactile'])

        if positive_source == 'visual':
            positive = random.choice(self.visual_embeddings[label])
            anchor = random.choice(self.tactile_embeddings[label])
        else:
            positive = random.choice(self.tactile_embeddings[label])
            anchor = random.choice(self.visual_embeddings[label])

        while True:
            negative_label = random.choice(self.labels)
            if negative_label != label:
                break

        negative_source = random.choice(['visual', 'tactile'])
        negative = random.choice(self.visual_embeddings[negative_label] if negative_source == 'visual' else self.tactile_embeddings[negative_label])

        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative), label


