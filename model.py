#Make the model now
class PowerPredictor(nn.Module):
  def __init__(self,input_features,num_outputs,neuronslist):
    super().__init__()
    self.input_features = input_features
    self.num_outputs = num_outputs
    self.neuronslist = neuronslist
    '''Here,
    input_features : Number of different features in dataset
    num_outputs : Number of outputs
    neuronslist : List of number of hidden units per layer
                  (Since there are three layers so there will be three elements in this list'''


    self.linear1 = nn.Linear(self.input_features,self.neuronslist[0])
    self.act1 = nn.Softplus()

    self.linear3 = nn.Linear(self.neuronslist[0], self.neuronslist[1])
    self.act3 = nn.Softplus()

    self.linear6 = nn.Linear(self.neuronslist[1], self.num_outputs)
    self.act6 = nn.Softplus()

  def forward(self,x):
    out = self.act1(self.linear1(x))
    #out = self.act2(self.linear2(out))
    out = self.act3(self.linear3(out))
    #out = self.act4(self.linear4(out))
    #out = self.act5(self.linear5(out))
    out = self.act6(self.linear6(out))
    return out
