# This script downloads and extracts data to a specified folder 

data_folder=data # TODO modify to where you want to store the data

mkdir $data_folder
cd $data_folder

echo "> Downloading datasets to $data_folder"

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
