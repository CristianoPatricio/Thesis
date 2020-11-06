----------------------------------------------------------------------
|		      Instructions for main.py			     |
----------------------------------------------------------------------

* All directory paths in main.py need to be replaced by others
considering the 'Data' folder, which contains all the necessary files 
except 'AwA-features.txt' due to its large size.

* To get the 'AwA-features.txt' file, click on the following
link: http://cvml.ist.ac.at/AwA2/AwA2-features.zip.

* All information about AwA2 data set is available at https://cvml.ist.ac.at/AwA2/.

----------------------------------------------------------------------
|	       Instructions for preprocessing_data.py	             |
----------------------------------------------------------------------

* Run <preprocessing_data.py> script to pickle the 'AwA-features.txt' file 
and then compress it into a file with extension '.pbz2'.

* You need to change the line 48 and add the correspondent path where your file 
is located to.

* After that run the script and the pickle file will be created in your current directory.
