from gtts import gTTS
import cv2
import numpy as np

def text_to_speech(text, filename):
    tts = gTTS(text)
    tts.save(f"{filename}.mp3")

def create_video(slides_content):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('presentation.avi', fourcc, 1, (640, 480))
    
    for slide_content in slides_content:
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(img, slide_content['body'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        video.write(img)
        text_to_speech(slide_content['body'], slide_content['title'])
    
    video.release()