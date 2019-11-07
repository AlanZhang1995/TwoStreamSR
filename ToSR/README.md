# Temporal-oriented SR

Our ToSR is implemented on Tensorflow and we provide our well-trained model in https://pan.baidu.com/s/17D2THJJ_uEWbeU5pYOrchg

Training Dataset
===================
Our training dataset is avaliable in https://pan.baidu.com/s/1Fd4UdiBfmK-38G9stL3Ngg. You can also use codes in './step1_training_dataset' dir to generate your training dataset (.h5).
* CDVL134 dataset is avaliable in https://pan.baidu.com/s/1VaMYHYDXdJtL_7JYk8lrSw
* After 'step2_genVideo2ExtractOF.m', you need to extract optical flow from obtained videos using tools we provided ('../Tools/video2frame&flow').
* To run 'step5_image2h5.m', you may need a computer with relatively large memory or do some adaptation.

Train and Test Code
===================
ToSR codes are mainly based on https://github.com/yhjo09/VSR-DUF. Since the author does not provide training codes, we implment the bilinear warp layer and rewrite the training process ourselves.
* Train code
	* Change the Training and validation dataset path on line 446-448 and 454-456 respectively in 'my1111.py' file. 
	* Use 'bash train_my1111.sh' to start training. Model will be save in 'TFoutput' dir.
* Test code
	* Change the LR video frame path on line 154 in 'test.py' file. 
	* Use 'bash run_test.sh' to start performing SR. Results will be save in 'results' dir.
	* After SR frame obtained, you can extract optical flow and feed both super-resolved RGB frames and optical flow maps into TSN or ST-Resnet to generate recognition accuracy (like SoSR).