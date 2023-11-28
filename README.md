# The-Big-3-project

General Purpose: Our programs will take x amount of faces that we want known from images, and save the faces attributes and a cutout of the faces into an analysis file and facial database, the programs will then open a live video feed and determine if the person that shows up in the feed is already in our known facial database, if they are they are skipped over and if they arent there face is extracted into a new facial database of unknown faces, once the stream is cut, the extracted faces are used to create another analysis file of the attributes of unknown faces.


Implementation Steps: 
  Run Face_Storage3ppl.py (you can put as many images as you want).
  Run create_known_faces_DB using the file path to your best faces.
  Run detectingandcomparing using the file path to the known faces extracted.
  Run create_unknown_faces_db with the file path to the unknownfaces extracted from detectingandcomparing.
Instructions on using our code:
1. Download all 4 py files [Face_Storage3ppl.py, create_known_faces_DB.py,detectingAndComparing.py, create_unknown_faces_db.py]
2. Install all necessary packages.
3. Select a group of input images you want to be the known faces.
4. Open Face_Storage3ppl.py and find the list variable “input_images”, and copy and paste the paths to each individual image. Run this code.
5. Open create_known_faces_DB.py, copy and paste the file path for the new images generated in the previous step in the list variable “best_faces_paths” and run this code.
6. You can now check the csv file named “face_analysis.csv”
7. Open detectingAndComparing.py and copy the file path for the folder generated in the Face_Storage3ppl.py [named “best_faces”]. Paste this file path in the variable “best_faces_path”. 
8. Change the threshold distance to whatever you want it to be. This is the maximum cosine distance allowed between two images. The default is set to 0.08
9. Run this code. It may take a second, but the camera should open, and if the threshold distance is tuned correctly, it should highlight your face. Press ‘q’ to stop running.
10. If you have extracted the unknown faces in the previous code, Open create_unknown_faces_db.py. Copy and paste the file path for the folder holding the extracted unknown faces in the variable “unrecognized_faces_folder” and run this code.
11. Open the csv file named “unrecognized_faces_analysis.csv” and if there were any extracted unknown faces, they should be displayed with their attributes.





Refernces:
https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
