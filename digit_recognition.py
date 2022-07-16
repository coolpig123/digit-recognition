import cv2
import pygame
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import os

# set up
dimensions = (560,560)
image = np.zeros(dimensions)
final_image = np.zeros((1,28,28))
clear_rect = pygame.Rect(0,670,285,100)
predict_rect = pygame.Rect(285,670,285,100)
window = pygame.display.set_mode((570, 770))
pygame.display.set_caption('digit recognition')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load trained model
model = keras.models.load_model("trained_models/digit_recognition")

# render text
pygame.font.init()
my_font = pygame.font.SysFont('lucidasans', 70)
text_surface_1 = my_font.render('clear', False, (255, 255, 255))
text_surface_2 = my_font.render('predict', False, (255, 255, 255))
text_surface_3 = my_font.render(f'prediction is ', False, (255, 255, 255))
active = True
while active:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            active = False
        if pygame.mouse.get_pressed()[0] and 5 < pygame.mouse.get_pos()[0] < 565 and 5 < pygame.mouse.get_pos()[1] < 545:
            pygame.draw.circle(window,(255,255,255),pygame.mouse.get_pos(),25,0)
        if clear_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
            window.fill((0, 0, 0, 0))
        if predict_rect.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
            # preprocess pixels
            for i in range(5, 565):
                for j in range(5, 565):
                    image[j-5][i-5] = window.get_at((i, j))[0] / 255.0

            # resize image
            final_image[0] = cv2.resize(image, dsize=(28, 28), interpolation=cv2.INTER_AREA)

            # make prediction
            prediction = model.predict(final_image)
            window.fill((0, 0, 0, 0))
            text_surface_3 = my_font.render(f'prediction is {np.argmax(prediction[0])}', False, (255, 255, 255))




    pygame.draw.rect(window, (0, 0, 255), (0, 0, 565, 5))
    pygame.draw.rect(window, (0, 0, 255), (0, 565, 565, 5))
    pygame.draw.rect(window, (0, 0, 255), (0, 0, 5, 570))
    pygame.draw.rect(window, (0, 0, 255), (565, 0, 5, 570))
    pygame.draw.rect(window, (0, 255, 0), (0, 670, 285, 100))
    pygame.draw.rect(window, (255, 0, 0), (285, 670, 285, 100))

    window.blit(text_surface_1, (60, 665))
    window.blit(text_surface_2, (310, 665))
    window.blit(text_surface_3,(40, 565))

    pygame.display.update()












