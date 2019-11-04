# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:47:05 2017

@author: Alan
"""

import os
import argparse

if __name__ == '__main__':
    '''
    命令行调用示例
    python genVideo_master.py "F:\python_genVideo\test" "F:\python_genVideo\res"
    '''
    parser = argparse.ArgumentParser(description='gen Video from png.')
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")
    
    args = parser.parse_args()
    path_input = args.src_dir
    path_output = args.out_dir
    #path_input ="F:\\python_genVideo\\test"
    #path_output ="F:\\python_genVideo\\res"
    #print path_input
    #print path_output
    flag=1 #跳过第一次
    for root,dirs,files in os.walk( path_input ):
        if flag==1:
            flag = 0
            continue

        #ffmpeg_input = os.path.join(root,'Frame%05d.png') #输入路径
        ffmpeg_input = os.path.join(root,'img_%05d.png') #输入路径
        ffmpeg_output= root.split('\\')[-1]              #输出文件名
        #print ffmpeg_input
        #print ffmpeg_output
        dirname=ffmpeg_output.split('_')[1]              #输出文件夹名（类名）
        path_class=os.path.join(path_output,dirname)     #输出文件夹路径
        isExists=os.path.exists( path_class )           
        # 判断结果
        if not isExists:
            os.makedirs(path_class)                       #没有就创建
        ffmpeg_output=os.path.join(path_class,ffmpeg_output)#输出路径
        if os.path.exists('{}.avi'.format(ffmpeg_output)):
		    continue
    
        ##fffmeg走起
        #cmd = 'cmd.exe /k ffmpeg -i {} -vcodec copy {}.avi'  #会卡黑框（每次都要关掉窗口）
        cmd = 'ffmpeg -i {} -vcodec copy {}.avi'.format(ffmpeg_input,ffmpeg_output)
        os.system(cmd)  