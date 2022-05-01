#!/bin/bash
output="/projects/p31689/Image_Analysis/output"
input="manifest.txt"

saved_time=$(date +"%Y%m%d%H%M%S")

package_path=$output/packaged_output/$saved_time
mkdir $package_path

while read path; do
  filename=$(basename $path)
  filename="${filename%.*}"
  output_path=$output/$filename/$(ls $output/$filename | tail -n 1)
  echo "Copying from $output_path..."
  echo " ...to $package_path/$filename"
  mkdir $package_path/$filename
  cp $output_path/data_labeled.csv $package_path/$filename
  cp $output_path/analysis.csv $package_path/$filename
  cp $output_path/*.png $package_path/$filename
  cp -r $output_path/mask_images $package_path/$filename
  cp -r $output_path/contoured_images $package_path/$filename
done < $input