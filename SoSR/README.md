# Spacial-oriented SR

Our SoSR is implemented with PyTorch and we provide our well-trained model in https://pan.baidu.com/s/17D2THJJ_uEWbeU5pYOrchg

Train
===================
The only change from ESRGAN to SoSR is the loss function (from MSE to WMSE). Since the author provides both training and testing code in https://github.com/xinntao/BasicSR. We only provide our revised parts as reference.

Test
===================
You may need several steps to test the recognition performance of our SoSR.
* Step1: Performing SR frame-by-frame
	* Use ESRGAN testing code (https://github.com/xinntao/ESRGAN) and our model to generate SR frames. 
* Step2: Synthesize SR videos
	* You can use Matlab or other tools to generate lossless videos from SR frames obtained in previous step. 
	* You can also use tools we provided in '../Tools/frame2video' dir to generate uncompressed SR videos.
* Step3: Generate recognition accuracy
	* Build the environment for TSN and ST-Resnet.
	* Extract frames (.jpg) and optical flow maps (.jpg) from SR videos using tools provided by TSN.
	* Run TSN and ST-Resnet to generate recognition accuracy following the steps recommanded by their author.