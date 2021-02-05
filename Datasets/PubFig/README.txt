---------------------------------------------------------------------------------------
|				     Instructions				      |
---------------------------------------------------------------------------------------

Required libraries:
-> keras (pip install keras)
-> keras_vggface (pip install keras_vggface)
-> opencv-python (pip install opencv-python)


1. Download cropped images of PubFig dataset from: http://www.sal.ipg.pt/user/1012390/datasets/PubFig/cropped_images.zip

2. Run the script 'extract_feats_im_crop.py' with '--images-dir' argument to extract image features.
	
	> python3 extract_feats_im_crop.py --images-dir '/path/to/cropped_images/'

3. Run the script 'make_feat_file.py'

	> python3 make_feat_file.py'

4. Run the script 'make_att_file_split.py' or 'make_att_file_50.py' (if you want to consider only the most populated 50 classes)

	> python3 'make_att_file_split.py'
	> python3 'make_att_file_50.py'

5. Run the script 'read_data.py' to get dataset statistics

	> python3 'read_data.py'

6. Feel free to evaluate the Celeb-A on two ZSL methods (Run the scripts 'ESZSL_CelebA.py' and 'SAE_CelebA.py')

	> python3 'SAE_PubFig.py'
	> python3 'ESZSL_PubFig.py' 
