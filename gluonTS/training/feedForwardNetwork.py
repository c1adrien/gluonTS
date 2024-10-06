
from models.feedForwardNetwork import FeedForwardNetwork
import lightning.pytorch as pl
import torch 

class LightningFeedForwardNetwork(FeedForwardNetwork, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"] #(batch size*time_series_len)
        future_target = batch["future_target"] #(batch size*prediction_len)

        assert past_target.shape[-1] == self.context_length
        assert future_target.shape[-1] == self.prediction_length

        distr_args, loc, scale = self(past_target) # appel directement le forward implemente ds le reseau de neurones. 
        distr = self.distr_output.distribution(distr_args, loc, scale) # distr contient distr.mean de taille (batch_size*horizon_prediction)
        #contient la standard deviation et sa shape (32,6)
        loss = -distr.log_prob(future_target) # (batch_size*horizon_projection). 
        #Calcul log vraisemblance des futures targets/aux params de distribution estimes..

        return loss.mean() #moyenne sur tout le batch size, et sur l'horizon de projection. 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

#TODO. faire evoluer sous la forme d'un trainer. Comme c'est le cas dans pytorch-ts. Après avoir regardé les détails de l'implémentation. 
#TODO. comprendre tout ce qui tourne autour des series --- 