# Useful Tools
Some tools maybe useful during experiment

frame2video
===================
You can use codes in './frame2video' dir to synthesize uncompressed videos from frames with ffmpeg.

* This code runs in Windows because it uses ffmpeg.exe. You can also it in Ubuntu with minor adaptation.  
* Use 'python genVideo_master.py SRC_FOLDER OUT_FOLDER' to run them in cmd or use 'run.bat' with some adaptation. 'SRC_FOLDER' is a source directory, the names of its subdirectory are the names of videos and their contents are frames of each video named as 'img_%05d.png'.
* Sometimes, you may need to do some adaptation on line 34 of 'genVideo_master.py' file to deal with different naming rules.

video2frame&flow
===================
You can use codes in './video2frame&flow' dir to split videos to frames ('.png') and optical flow ('.mat'). We also provide function to read flow in matlab, readMatFlow.m.

* These codes are adapted from https://github.com/yjxiong/dense_flow
* You need to run 'bash build_all.sh' to make them first.
* As a test, we use commands in 'build_all2.sh' to make these codes successfully using environment of docker image 'cuda8-cudnn6-dev-ubuntu16.04' provided by Nvidia. You can adapt 'build_all2.sh' to a dockerfile yourself.
* After making these code, you can use 'bash Extract_Frames_flow.sh' to run them with some adaptation.