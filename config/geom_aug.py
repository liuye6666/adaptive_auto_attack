default_dataset='cifar10'#used for selecting AAA hyperparameters
dataset='cifar10'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

sstransformation=dict(
    max_r=30., 
    max_t=10., 
    max_s=1.2248, #this scale allows a max rotation of 15
    shape=(32,32,3),
)