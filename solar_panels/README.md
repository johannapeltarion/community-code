Follow the instructions below to preprocess the dataset for the "Predicitive maintenance of solar panels" training example. For further information about the dataset, see the zae-bayern/elpv-dataset repository. Please note that the data set is subject to separate third party license terms, please see disclaimer below.

# Preparing the dataset
1. Download the dataset.
2. Run solar_preprocess.py:  
`
python solar_preprocess.py [<option>=True,False,...] {<your_path>/elpv-dataset}
`\
The preprocessed dataset file, preprocessed.zip, will be created in the same folder as the dataset.\
Options:\
--rotate Adds duplicated images rotated by 90 degrees\
--stratify_on_type Stratifies subsets on both module type and defect probability\
--image_as_np Converts images to NPY\
--image_as_vgg19 Converts images to NPY and normalizes for VGG-19\
--balance Upsamples 100% defects (x2)\
--target_size Sets image size (default: 300 px)

# Analysis of the deployed model
1. Note the URL and token in the Deployment view
2. Run solar_analysis.py:  
`
python solar_analysis.py {<dataset_path>} {url} {token}
`  
The generated CSV file will contain the actual and predicted values for each image in the validation subset. You may use this data for analysis of the model in notebooks etc.\
\
\
\
*Please note that data sets, models and other content, including open source software, (collectively referred to as “Content”) provided and/or suggested by Peltarion for use in the Platform, may be subject to separate third party terms of use or license terms. You are solely responsible for complying with the applicable terms. Peltarion makes no representations or warranties about Content and specifically disclaim all responsibility for any liability, loss, or risk, which is incurred as a consequence, directly or indirectly, of the use or application of any of the Content.*
