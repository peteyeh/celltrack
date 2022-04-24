import cv2
import os
import re

base_path = "/projects/b1042/Abazeed_Lab/Pete_Priyanka/source_images"
output_dir = "/projects/b1042/Abazeed_Lab/Pete_Priyanka/merged_images"
sub_dir_pattern = "[0-9]{6}_[0-9]{6}_Plate 1"
tif_pattern = "([A-H][1]?[0-9])_[0-9]{2}_[0-9]_[0-9]_Bright Field_([0-9]{3}).tif"
tif_sub_pattern = "_[0-9]{2}_[0-9]_[0-9]_Bright Field_[0-9]{3}.tif"

root_dirs = os.listdir(base_path)
plates = [s.split('_')[-1] for s in root_dirs]
for i in range(len(root_dirs)):
    for sub_dir in os.listdir(os.path.join(base_path, root_dirs[i])):
        if re.match(sub_dir_pattern, sub_dir):
            root_dirs[i] = os.path.join(root_dirs[i], sub_dir)

max_runs = max([int(p.split('-')[1]) for p in plates if len(p.split('-')) == 2])

mapping = {}
for i in range(len(plates)):
    if len(plates[i].split('-')) == 1:
        mapping[int(plates[i])] = [root_dirs[i],]

for r in range(1, max_runs+1):
    for i in range(len(plates)):
        s = plates[i].split('-')
        if len(s) == 2 and int(s[1]) == r:
            mapping[int(s[0])] += [root_dirs[i],]

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Deliberately processing one image stack at a time here, to avoid out of memory issues
for plate in mapping.keys():
    image_stacks = set()
    files = os.listdir(os.path.join(base_path, mapping[plate][0]))
    for file in files:
        result = re.match(tif_pattern, file)  # ignore non-bright-field images
        if result:
            image_stacks.add(result.group(1))
    for image_stack in image_stacks:
        image_paths = []
        for i in range(len(mapping[plate])):
            if i >= 0:
                files = os.listdir(os.path.join(base_path, mapping[plate][i]))
            full_pattern = image_stack+tif_sub_pattern
            image_paths += \
                sorted([os.path.join(mapping[plate][i], p) for p in filter(re.compile(full_pattern).match, files)])
        image_array = []
        for image_path in image_paths:
            path = os.path.join(base_path, image_path)
            image_array += [cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE),]
        print("Writing %s." % (str(plate)+image_stack+".tiff"))
        cv2.imwritemulti(os.path.join(output_dir, str(plate)+image_stack+".tiff"), image_array)
