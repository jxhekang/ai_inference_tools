# from torchstat import stat
import torchstat
import torchvision.models as models
model = models.alexnet()
torchstat.stat(model, (3, 224, 224))