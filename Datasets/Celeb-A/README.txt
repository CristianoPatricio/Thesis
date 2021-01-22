------------------------------------------------------------------------------------
|				HOW TO USE FILES				   |
------------------------------------------------------------------------------------

1. Download raw images (img_align_celeba.zip ~ 1.3GB) from: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing

2. Donwload Celeb-A VGG16 features (~ 267MB) from: http://www.sal.ipg.pt/user/1012390/datasets/Celeb-A/celeba_feats_512.zip

3. All other necessary files are in the 'data' folder

4. Make sure you change the paths in all python scripts accordingly to your working directory

5. Run the script 'make_vgg16_file.py'

6. Run the script 'make_att_split_file.py' or 'make_att_split_50_classes.py' (if you want to consider only 50 classes)

7. Run the script 'read_data.py' to get the dataset statistics

8. Feel free to evaluate the Celeb-A on two ZSL methods (Run the scripts 'ESZSL_CelebA.py' and 'SAE_CelebA.py')
