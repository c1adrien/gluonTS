from typing import List, Optional, Callable, Iterable
import torch
import torch.nn as nn
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast_generator import DistributionForecastGenerator

"""
This part implement a simple version of """


def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class FeedForwardNetwork(nn.Module):
    """ Custom neural network. i.e. 
        preidction_length : taille de la prédiction 
        context_length : longueur de la fenetere que le modele prend pour faire sa prediction
        hidden_dimension : specifie le nombre de couches cachees et le nombre de neurones par couches ici
        distr_output : pour l'incerttude des predictions, le modele apprend donc a predire une tStudent. On pourrait prendre gaussienne
        batch_norm : normalisation par batch apres chaque couche lineaire. ça accelere le training 
        scaling : methode que l'on appelle pour renormaliser les donnees d entrees. """
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimensions: List[int],
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
        scaling: Callable = mean_abs_scaling,
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling

        dimensions = [context_length] + hidden_dimensions[:-1]

        # ------ modules ----- 
        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            self.__make_lin(dimensions[-1], prediction_length * hidden_dimensions[-1])
        )

        self.nn = nn.Sequential(*modules) #combines toutes les couches de module en une unique sequence. 
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1]) #on prend les outputs et on projette. objet de type PtArgProj
        # on se connecte à la derniere couche et on projette. 

    @staticmethod #fonction ordinaire incluse dans la classe pour des raisons d'organisation. 
    #ne modifie pas ou n'interagit pas avec les instances/les attributs de la classe. 
    def __make_lin(dim_in, dim_out):
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def forward(self, past_target):
        #passage dans le reseau de neurones. renvoie une distribution de sortie. Pour les copules, comment ça va se traduire ? 
        scale = self.scaling(past_target) #past target est de size (batch size* time_series_length)
        scaled_past_target = past_target / scale
        nn_out = self.nn(scaled_past_target) #(batch_size*150). Or self.hidden_dimensions[-1] = 25. (derniere couche precisee...)
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped) #PtArgProj : renvoie les qrgs de la distribution apres avoir passé ds l'objet PtArgProj
        return distr_args, torch.zeros_like(scale), scale # distr_args, loc, scale. Les 3 éléments. renvoie les parametres de la distribution + 2 tenseurs supplémentaires pour mise à l'échelle. 

    def get_predictor(self, input_transform, batch_size=32):
        #permet de faire des predictions apres entrainement du reseau. Batchsize=32 : le modele va traiter les data par lot de 32.
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
        )
    

"""class PtArgProj(nn.Module):
    def __init__(self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]]):
        super().__init__()
        self.args_dim = args_dim
        self.proj = nn.ModuleList(
            [nn.Linear(in_features, dim) for dim in args_dim.values()]
        )
        self.domain_map = domain_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        params_unbounded = [proj(x) for proj in self.proj]
        return self.domain_map(*params_unbounded)"""