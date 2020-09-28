import numpy as np

"""In this file we simulate data samples and labels for HCAE

	simulate_data(n, subjects)
	Input
		n: (integer) the number of nodes in a data sample
		subjects: (integer) the total number of samples
	Output
		samples: (subjects x n x n) contains the simulated datasamples


	simulate_labels(ones, subjects)
	Input
		ones: (integer) the number of labels which will be assigned '1'
		subjects: (integer) the number of labels in dataset"""

def simulate_data(n, subjects):
    
    samples = []
    for i in range(subjects):
        mean, std = np.random.rand(), np.random.rand()
        b = np.abs(np.random.normal(mean, std, (n,n))) % 1.0
        b_symm = (b + b.T)/2
        b_symm[np.diag_indices_from(b_symm)] = 0
        samples.append(b_symm)
    

    return np.asarray(samples)

def simulate_labels(ones, subjects):
    labels = np.zeros(subjects)
    labels[:ones]  = 1
    np.random.shuffle(labels)
    return np.asarray(labels)
            
