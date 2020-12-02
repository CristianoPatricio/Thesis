----------------------------------------------------------------------
|	       Instructions for download_AWA2_dataset.py	     |
----------------------------------------------------------------------

1. Run <download_AWA2_dataset.py> script to download and extract the 'AWA2' or 'AWA2_mat' dataset.
	
	> $python3 download_AWA2_dataset.py -d [AWA2, AWA2_mat]

After perform the step 1. you will be able to see the 'AWA2' folder in your current directory.

2. To make use of the AWA2 data you should import the load_awa2_data() function in your script:
	> from AWA2.awa import load_awa2_data
	or
	> from AWA2_mat.preprocessing import load_awa2_data
	
3. If you download the 'AWA2' dataset, please follow the instructions in the section 3.1, otherwise 
follow the instructions in the section 3.2.

3.1 [AWA2] 
The return of the load_awa2_data(attributes, split) function is a dictionary. 
The parameters {attributes, split} refers to the type of required attributes = {'continuous', 'binary'} 
and the desired split = {'ss', 'ps'}, where 'ss' denotes for 'standard split' and 'ps' denotes for 'proposed slipt'
as introduced in [1]. 

To access the required train, val and test data you can do it by typing:

	awa_data = load_awa2_data(attributes='continuous', split='ps')

	# Training data
	X_train = awa_data['X_train']
	y_train = awa_data['y_train']
	# Semantic attributes
	s_train_per_class = awa_data['S_train_per_class']
    	s_train_per_sample = awa_data['S_train_per_sample'] 
    	
	
	# Validation data
	X_val = awa_data['X_val'] 
    	y_val = awa_data['y_val']
    	# Semantic attributes
    	s_val_per_class = awa_data['S_val_per_class'] 
    	s_val_per_sample = awa_data['S_val_per_sample'] 
    
	
	# Test data
	X_test = awa_data['y_test'] 
	y_test = awa_data['X_test'] 
	s_test_per_class = awa_data['S_test_per_class'] 
	s_test_per_sample = awa_data['S_test_per_sample'] 
	
-----------------------------------------------------------------------
	
3.2 [AWA2_mat] 
The return of the load_awa2_data(attributes) function is a dictionary. 
The parameters attributes refers to the type of required attributes = {'continuous', 'binary'}.

To access the required train, val and test data you can do it by typing:

	awa_data = load_awa2_data(attributes='continuous')

	# Training data
	X_train = awa_data['X_train']
	y_train = awa_data['y_train']
	s_train = awa_data['S_train']

	# Validation data
	X_val = awa_data['X_val']
	y_val = awa_data['y_val']
	s_val = awa_data['S_val']

	# Test data
	X_test_seen = awa_data['X_test_seen']
	y_test_seen = awa_data['y_test_seen']
	s_test_seen = awa_data['S_test_seen']
	X_test_unseen = awa_data['X_test_unseen']
	y_test_unseen = awa_data['y_test_unseen']
	s_test_unseen = awa_data['S_test_unseen']

As you can see, the 'AWA2_mat' option provides you a test set that includes both seen and unseen classes 
to evaluate in the generalized zero-shot learning setting.
	
-----------------------------------------------------------------------
| 				REFERENCES			      |													
-----------------------------------------------------------------------

[1] Xian, Yongqin and Lampert, H. Christoph and Schiele, Bernt and Akata, Zeynep. 
Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly. TPAMI, 2018.
