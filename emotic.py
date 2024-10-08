import torch
import torch.nn as nn

class Emotic(nn.Module):
    ''' Emotic Model with face feature integration'''
    def __init__(self, num_context_features, num_body_features, num_face_features):
        super(Emotic, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features  # Ensure this is correctly defined

        # Combine context, body, and face features
        self.fc1 = nn.Linear((self.num_context_features + self.num_body_features + self.num_face_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, x_context, x_body, x_face):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        face_features = x_face.view(-1, self.num_face_features)
        
        fuse_features = torch.cat((context_features, body_features, face_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        
        return cat_out, cont_out
