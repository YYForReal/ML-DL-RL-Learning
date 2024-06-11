import torch.nn as nn
from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss
from nbdt.hierarchy import generate_hierarchy
import pretrainedmodels

model = pretrainedmodels.__dict__['fbresnet152'](num_classes=1000, pretrained='imagenet')

# 1. generate hierarchy from pretrained model
generate_hierarchy(dataset='Imagenet1000', arch='fbresnet152', model=model)

# 2. Fine-tune model with tree supervision loss
criterion = nn.CrossEntropyLoss()
criterion = SoftTreeSupLoss(dataset='Imagenet1000', hierarchy='induced-fbresnet152', criterion=criterion)

# 3. Run inference using embedded decision rules
model = SoftNBDT(model=model, dataset='Imagenet1000', hierarchy='induced-fbresnet152')