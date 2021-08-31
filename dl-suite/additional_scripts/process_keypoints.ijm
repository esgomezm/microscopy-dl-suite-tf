// Specify the directory with the images to process 
path_images = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/test/stack2im/keypoints/";
path_results = "/home/esgomezm/Documents/3D-PROTUCEL/data/test/stack2im/keypoints_prob/";
//path_results = "/home/esgomezm/Documents/3D-PROTUCEL/data/train/stack2im/keypoints_prob/";

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
		setThreshold(2, 6);
		setOption("BlackBackground", true);
		run("Convert to Mask");
		run("Gray Morphology", "radius=3 type=circle operator=dilate");
		run("Distance Map");
		run("32-bit");
		run("Gaussian Blur...", "sigma=1");
		getStatistics(_, _, min, max);		
		print(max);
		if (max > 0){
			run("Divide...", "value=" + max);
		}
		rename(image_name);
		saveAs("Tiff", path_results+image_name+".tif");
		close("*");
	}
}