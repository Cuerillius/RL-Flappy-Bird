import os
import pygame

sprites = {}
audios = {}


def load_sprites():
    path = os.path.join("assets", "sprites")
    for file in os.listdir(path):
        sprites[file.split('.')[0]] = pygame.image.load(
            os.path.join(path, file))


def get_sprite(name):
    return sprites[name]


def load_audios():
    try:
        path = os.path.join("assets", "audios")
        for file in os.listdir(path):
            audios[file.split('.')[0]] = pygame.mixer.Sound(
                os.path.join(path, file))
    except Exception as e:
        pass


def play_audio(name):
    try:
        audios[name].play()
    except Exception as e:
        pass
