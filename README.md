# The-Big-3-project

General Purpose: Our programs will take x amount of faces that we want known from images, and save the faces attributes and a cutout of the faces into an analysis file and facial database, the programs will then open a live video feed and determine if the person that shows up in the feed is already in our known facial database, if they are they are skipped over and if they arent there face is extracted into a new facial database of unknown faces, once the stream is cut, the extracted faces are used to create another analysis file of the attributes of unknown faces.


Implementation Steps: Run Face_Storage3ppl.py (you can put as many images as you want)
Run create_known_faces_DB using the file path to your best faces
Run detectingandcomparing using the file path to the known faces extracted 
Run create_unknown_faces_db with the file path to the unknownfaces extracted from detectingandcomparing





Refernces:
https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/
