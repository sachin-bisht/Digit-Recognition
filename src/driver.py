import mnist_loader as mload
import training_and_result as tAr


def main():
	images,labels = mload.mnist_loader()

	weight1, weight2 = tAr.training(images, labels)

	images, labels = mload.mnist_loader(False)
	tAr.judgement(images, labels, weight1, weight2)


if (__name__ == "__main__"):
	main()
