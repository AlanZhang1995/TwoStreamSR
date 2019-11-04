# SR for Action Recognition

Two-Stream Action Recognition-Oriented Video Super-Resolution (ICCV2019)

License and Citation
===================

Content
===================
This code consists of Three parts:
* Training and Testing code of our Spacial-oriented SR
	* SoSR is implemented with Pytorch based on ESRGAN (https://github.com/xinntao/BasicSR)
	* We provide our well-trained caffemodel in './SoSR/test/step1_SR_each_frame' dir
* Training and Testing code of our Temporal-oriented SR
	* ToSR is implemented with Tensorflow based on VSR-DUF (https://github.com/yhjo09/VSR-DUF)
	* We provide our well-trained model in './ToSR2/step2_train&test/test/TFoutput' dir
* Some tools maybe useful during experiment
	* Tools for synthesizing videos from frames without compression using ffmpeg
	* Tools for spliting videos to frames ('.png') and calculating optical flow ('.mat')
	
By the way, we use Temporal Segment Network (TSN) and Spacial-Temporal Residual Network (ST-Resnet) to generate recognition accuracy. Please refer to https://github.com/yjxiong/temporal-segment-networks and https://github.com/feichtenhofer/st-resnet for their code respectively. 
