# bimap_registration

Overview of three registration approaches.

* ants_register.py: uses the ants library to register each subsequent frame to the first frame
* lddmm_register.py: applys lddmm mapping fir registration
* mc_cotracker.py: uses cotracker to derive velocity field, which is used to warp all frames onto the first frame
* input/: stores data which needs to be registered
* output/: folder to store registered output
* cotracker/: everything stored to run cotracker cloned from [CoTracker Repository](https://github.com/facebookresearch/co-tracker.git)

Data is stored in the 
All three scripts share the same main method by defining the following:

* Define video path:
  ````
  video_file = "/home/rzlin/va11moje/co-tracker/input/Drosophila_Ca_bad.avi"
  ````

* Apply registration
  ````
  result = lddmm_registration(frames)
  ````
  
* Save result with 15 frames per second
  ````
   save_and_display_video(np.array(result), f'output/{filename}_lddm_registered.mp4', fps=15)
  ````
