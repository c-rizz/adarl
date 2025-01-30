import torch as th
from autoencoding_rl.utils import build_mlp_net

from dataclasses import dataclass
from typing import List

class RNDNoveltyEstimator(th.nn.Module):

    @dataclass
    class Hyperparams:
        nn_arch : List[int] = None
        input_size : int = None
        feature_vec_size : int = None
        learning_rate : float = None
        ensemble_size : int = None

    def __init__(self, nn_arch : List[int], input_size : int, feature_vec_size : int, learning_rate):
        super().__init__()
        self._hyperparams = self.Hyperparams()
        self.set_hyperparams(nn_arch, input_size, feature_vec_size, learning_rate, 5)
        self.build_models()

    def set_hyperparams(self, nn_arch : List[int], input_size : int, feature_vec_size : int, learning_rate : float, ensemble_size : int):
        self._hyperparams.nn_arch = nn_arch
        self._hyperparams.input_size = input_size
        self._hyperparams.feature_vec_size = feature_vec_size
        self._hyperparams.learning_rate = learning_rate
        self._hyperparams.ensemble_size = ensemble_size

    def _build_net(self):
        return build_mlp_net(arch = self._hyperparams.nn_arch, 
                            input_size = self._hyperparams.input_size,
                            output_size = self._hyperparams.feature_vec_size,
                            ensemble_size=self._hyperparams.ensemble_size,
                            last_activation_class=th.nn.Tanh,
                            return_ensemble_mean=False,
                            hidden_activations=th.nn.LeakyReLU,
                            return_ensemble_std=False)

    def build_models(self):
        self._target_net = self._build_net()
        for p in self._target_net.parameters(): # apparently requires_grad_ doesn't work on scriptmodules
            p.requires_grad_(False)
        self._predictor_net = self._build_net()

        self._optimizer = th.optim.Adam(self._predictor_net.parameters(), lr = self._hyperparams.learning_rate)
        self._optimizer.zero_grad()

    def forward(self, batch : th.Tensor):
        with th.no_grad():
            target_features = self._target_net(batch)
        predicted_features = self._predictor_net(batch)
        # we now have two [batch_size, ensemble_size, feature_size] tensors
        # we do the mean across both feature_size and ensemble_size.
        #     as the diffs are squared ensembles cannot compensate each other
        # we return a [batch_size] tensor. i.e. we return the novelty for each sample
        return th.mean(th.square(target_features-predicted_features),dim=(1,2)) 


    def train_model(self, batch : th.Tensor):

        self.train() # Put module in train mode
        self._optimizer.zero_grad(set_to_none=True)
        square_errors = self(batch)
        loss = th.mean(square_errors)
        loss.backward()
        self._optimizer.step()
        return loss