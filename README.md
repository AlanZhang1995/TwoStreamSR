# SR for Action Recognition

Two-Stream Action Recognition-Oriented Video Super-Resolution (ICCV2019)

License and Citation
===================

Content
===================
This code consists of Three parts:
* Training and Testing code of our Spacial-oriented SR
	* SoSR is implemented with PyTorch based on ESRGAN (https://github.com/xinntao/BasicSR)
	* Our well-trained model can be found in https://pan.baidu.com/s/17D2THJJ_uEWbeU5pYOrchg
* Training and Testing code of our Temporal-oriented SR
	* ToSR is implemented with TensorFlow based on VSR-DUF (https://github.com/yhjo09/VSR-DUF)
	* Please put our well-trained model (the folder 'VSR-DUF' as well as the files inside it) into './ToSR/step2_train&test/test/TFoutput/model' dir
* Some tools maybe useful during experiment
	* Tools for synthesizing videos from frames without compression using ffmpeg
	* Tools for spliting videos to frames ('.png') and calculating optical flow ('.mat')
	
By the way, we use Temporal Segment Network (TSN) and Spacial-Temporal Residual Network (ST-Resnet) to generate recognition accuracy. Please refer to https://github.com/yjxiong/temporal-segment-networks and https://github.com/feichtenhofer/st-resnet for their code respectively. 
