#Making the input data a Pytorch Dataset
class PowerDataset(Dataset):
  def __init__(self,x,y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    #Make the input data a torch tensor
    self.x = torch.from_numpy(x)
    self.y = torch.from_numpy(y)

  def __getitem__(self,index):
    return self.x[index],self.y[index]

  def __len__(self):
    return self.x.shape[0]
