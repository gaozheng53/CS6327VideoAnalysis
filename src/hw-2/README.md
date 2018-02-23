## Description
Video Analysis Assignment-2
- Detect a white ball in the video, we need to ignore the area outside the table, it's a little bit tricky. And draw a 
white line in an image.
- More details in the file`CS 6327 Video Analytics Assignment 2.doc`
- Source code is `Assignment.py`
- Source video is `cs6327-a2.mp4`
- Output path tracking image is `path.jpg`

Complete Time: 2/23/2018
Author: Zheng Gao(zxg170430)

## How To Run
Put source file in the same directory of source code, if you modified source video's postion or name, you need to 
modify code line 15 `camera = cv2.VideoCapture('cs6327-a2.mp4')` to your new name and its relative position.
Then just run  `Assignment.py`.    
Enjoy~ :)

## Want to Fix

- Bonus point assignment: convert speed to real-world speed.
- Hard-coded about circle's radius around the ball.
