import numpy as np
import csv
import h5py
import os 
import argparse

INTERIM_DIR = "./data/interim"
PROCESSED_DIR = "./data/processed"


def calculate_coverage_statistics(dataset):
	if os.path.exists(os.path.join(PROCESSED_DIR, f"{dataset}_coverage_widths.csv")):
		raise IOError("This csv file already exists")

	with h5py.File(os.path.join(INTERIM_DIR,f"{dataset}_credible_set.hdf5"), 'r') as credible_file:
	  with open(os.path.join(PROCESSED_DIR, f"{dataset}_coverage_widths.csv", 'a') as csvfile:
	    fieldnames = ['method', 'split', 'eps', 'coverage', 'width']
	    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	    writer.writeheader()

	    if dataset == 'imagenet':
		    for method in list(credible_file.keys()):
		      for split in list(credible_file[method].keys()):
		        labels = credible_file[method][split]['labels'][...][:, np.newaxis]
		        labels_set = np.apply_along_axis(expand_labels, 1, labels)

		        for eps in [.05]: 
		          print(method, split, eps)
		          
		          credible_sets = credible_file[method][split][str((1-eps)*100)+'_credible_set']

		          
		          coverage = np.mean(np.sum(np.multiply(credible_sets, labels_set), axis=-1))


		          width = np.mean(np.sum(credible_sets, axis=-1), axis=-1)

		          writer.writerow({'method':method, 'split':split, 'eps':eps, 'coverage': coverage, 'width': width})
		if dataset != 'imagenet':
			for method in list(credible_file.keys()):
		      for split in list(credible_file[method].keys()):
		        labels = np.expand_dims(credible_file[method][split]['labels'], -1)
		        labels_set = np.apply_along_axis(expand_labels, -1, labels)

		        for eps in [.05]: 
		          print(method, split, eps)
		          
		          credible_sets = credible_file[method][split][str((1-eps)*100)+'_credible_set']
		          
		          coverages = np.mean(np.sum(np.multiply(credible_sets, labels_set), axis=-1), axis=-1)

		          widths = np.mean(np.sum(credible_sets, axis=-1), axis=-1)

		          for coverage, width in zip(coverages, widths):
		            writer.writerow({'method':method, 'split':split, 'eps':eps, 'coverage': coverage, 'width': width})


def main(): 
	parser = argparse.ArgumentParser("Create a credible set from Ovadia et al predictions")
	parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar', 'imagenet'])
	args = parser.parse_args()

	create_credible_set_h5py(args.dataset)