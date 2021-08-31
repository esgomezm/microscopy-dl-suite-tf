// Specify the directory with the images to process 
path_images = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/DATA/data_corrected/test/stack2im/labels/";
path_results = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/DATA/data_corrected/test/stack2im/contours/";

path_images = "/home/esgomezm/Documents/3D-PROTUCEL/data/CTC/FluoC3DLMDA231/stack2im/labels/";
path_results = "/home/esgomezm/Documents/3D-PROTUCEL/data/CTC/FluoC3DLMDA231/stack2im/contours/";

path_images = "/home/esgomezm/Documents/3D-PROTUCEL/keypoints_data/train/stack2im/labels2/";
path_results = "/home/esgomezm/Documents/3D-PROTUCEL/keypoints_data/train/stack2im/contours/";

path_images = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/train/stack2im/labels/";
path_results = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/train/stack2im/contours/";

// Read the name of all the files in the image and ground truth directories.
list = getFileList(path_images);
print(list.length+" images to evaluate.");

for (i=0; i<list.length; i++) {
	if (!endsWith(list[i], "/")){
		// store the name of the image to save the results
		image_name = split(list[i], ".");
		image_name = image_name[0];
		// open the image
		open(path_images + list[i]);  
		run("Label Boundaries");
		rename(image_name);
		saveAs("Tiff", path_results+image_name+".tif");
		close("*");
	}
}
