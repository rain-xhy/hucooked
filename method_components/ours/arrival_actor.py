"""Arrival forecasting and the action-score residual used by Ours."""

import torch
from torch import nn
from torch.nn import functional as F

from acac.acac_marl.cores.acac.models import AgentCentricGRUActor


class ArrivalActor(AgentCentricGRUActor):
    def forward(self, x, h=None, eps=0.0, test_mode=False, time_emb=None):
        public_obs = x
        if time_emb is not None:
            x = torch.cat((x, self.time_embedding(time_emb)), dim=-1)
        history, h = self.encoder(x, h)
        features = F.leaky_relu(self.fc3(history))
        if self.use_ln:
            features = self.ln(features)
        arrival = self.arrival_head(torch.cat((public_obs, history.detach()), dim=-1)).sigmoid()
        scores = self.fc4(features) + self.arrival_residual(torch.cat((features, arrival), dim=-1))
        logp = F.log_softmax(scores, dim=-1)
        if not test_mode and eps:
            mixed = (1-eps)*logp.exp() + eps/logp.size(-1)
            logp = mixed.log()
        return logp, h

def forecast_network(obs, history):
    return nn.Sequential(nn.Linear(obs+history, 48), nn.ReLU(), nn.Linear(48, 12))

def attach(actor, initializer=None):
    width = actor.encoder.gru.hidden_size
    obs = actor.encoder.fc1.in_features
    if actor.use_time_emb:
        obs -= actor.time_embedding.embedding_dim
    actor.__class__ = ArrivalActor
    device = actor.fc4.weight.device
    actor.arrival_head = forecast_network(obs,width).to(device)
    if initializer is not None:
        actor.arrival_head.load_state_dict(initializer)
    actor.arrival_head.requires_grad_(False)
    actor.arrival_residual = nn.Linear(actor.fc4.in_features+12,actor.fc4.out_features).to(device)
    nn.init.zeros_(actor.arrival_residual.weight)
    nn.init.zeros_(actor.arrival_residual.bias)
    return actor
