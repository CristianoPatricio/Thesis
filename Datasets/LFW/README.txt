---------------------------------------------------------------------------------------
|				     Instructions				      |
---------------------------------------------------------------------------------------

Required libraries:
-> keras (pip install keras)
-> keras_vggface (pip install keras_vggface)
-> opencv-python (pip install opencv-python)


1. Download cropped images of LFW dataset from: https://conradsanderson.id.au/lfwcrop/lfwcrop_color.zip

2. Run the script 'extract_features.py'
	
	> python3 extract_features.py

3. Run the script 'make_feat_file.py'

	> python3 make_feat_file.py'

4. Run the script 'make_att_file_split.py' or 'make_att_file_50.py' (if you want to consider only the most populated 50 classes)

	> python3 'make_att_file_split.py'
	> python3 'make_att_file_50.py'

5. Run the script 'read_data.py' to get dataset statistics

	> python3 'read_data.py'

6. Feel free to evaluate the Celeb-A on two ZSL methods (Run the scripts 'ESZSL_CelebA.py' and 'SAE_CelebA.py')

	> python3 'SAE_LFW.py'
	> python3 'ESZSL_LFW.py' 
