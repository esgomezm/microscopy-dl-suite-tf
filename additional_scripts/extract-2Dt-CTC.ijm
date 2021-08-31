// Specify the directory with the images to process 
path_images = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/CTC/Fluo-C3DL-MDA231/02/";
path_silver = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/CTC/Fluo-C3DL-MDA231/02_ST/SEG/";
path_results = "/media/esgomezm/sharedisk/Documents/BiiG/3D-PROTUCEL/data/CTC/Fluo-C3DL-MDA231/stack2im";


// Converts 'n' to a string, left padding with zeros
// so the length of the string is 'width'
function leftPad(n, width) {
      s =""+n;
      while (lengthOf(s)<width)
          s = "0"+s;
      return s;
  }

zplane = 8;
// Read the name of all the files in the image and ground truth directories.
list = getFileList(path_images);
print(list.length+" 3D time points.");
for (i=0; i<list.length; i++) {
	print(list[i]);
	if (!endsWith(list[i], "/")){
		// store the name of the image to save the results		
		image_name = "t" + leftPad(i, 3) + ".tif";
		st_name = "man_seg" + leftPad(i, 3) + ".tif";
		// open the image
		open(path_images + image_name);  
		open(path_silver + st_name);  
		selectWindow(image_name);
		run("Make Substack...", "  slices="+zplane);
		rename("input");
		selectWindow(st_name);
		run("Make Substack...", "  slices="+zplane);
		rename("output");
		close(image_name);
		close(st_name);
		print(i);
		if (i==0){
			selectWindow("input");
			rename("2D-video");
			selectWindow("output");
			rename("2D-video-output");
		} else {
			run("Concatenate...", "title=2D-video open image1=2D-video image2=input image3=[-- None --]");
			run("Concatenate...", "title=2D-video-output open image1=2D-video-output image2=output image3=[-- None --]");
		}
		
	}
}
