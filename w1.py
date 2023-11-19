from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from gtts import gTTS
import pygame
import keyboard  
import os
import cv2

model = VisionEncoderDecoderModel.from_pretrained('vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('vit-gpt2-image-captioning')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

def predict_step(image):
    pixel_values = feature_extractor(
        images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

def caption_and_play(image_path):
    image = Image.open(image_path)
    predcap = predict_step(image=image)
    print(predcap)
    
    try:
        tts = gTTS(text=predcap, lang='en')
        tts.save('temp.mp3')
        pygame.mixer.init()
        pygame.mixer.music.load('temp.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        pygame.mixer.quit()
        os.remove('temp.mp3')

    except Exception as e:
        print("Error:", e)

def on_key_press(e):
    if e.event_type == keyboard.KEY_DOWN:
        cap = cv2.VideoCapture(2)
        ret, frame = cap.read()
   
    
        if ret:
            photo_filename = f"testimg.jpg"
            cv2.imwrite(photo_filename, frame)
            print(f"Photo saved as {photo_filename}")
        caption_and_play(photo_filename)

keyboard.hook(on_key_press)
keyboard.wait("esc") 
