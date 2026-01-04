import torch.nn as nn
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Model.RPGenerator import GaussianRP
from Model.PredictionHead.RiskPredictor import RiskPredictor

class Autoencoder(nn.Module):
    def __init__(self, emb_dim=16, dropout=0.1):
        super().__init__()
        self.rp_generator = GaussianRP()
        self.encoder = Encoder(emb_dim=emb_dim, dropout=dropout)
        self.decoder = Decoder(emb_dim=emb_dim, dropout=dropout)
        self.risk_predictor = RiskPredictor(emb_dim=emb_dim, output_dim=1)

    def forward(self, x):
        x = self.rp_generator(x)
        latent = self.encoder(x)
        reconstructed_de = self.decoder(latent)
        risk = self.risk_predictor(latent)
        return reconstructed_de, x, latent, risk