from mnist import MNIST

def mnist_loader(value = True):
	mndata = MNIST('./')
	if value == True:
		images, labels = mndata.load_training()
	else:
		images, labels = mndata.load_testing()

	return images, labels
