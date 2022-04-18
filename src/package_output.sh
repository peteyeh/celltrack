#!/bin/bash
output="/projects/b1042/Abazeed_Lab/Pete_Priyanka/output"
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
  cp $output_path/*.png $package_path/$filename
  cp -r $output_path/mask_images $package_path/$filename
done < $input