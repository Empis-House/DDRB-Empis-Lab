# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:10:23 2023

@author: Programmed with LLM
"""

from PIL import Image, ImageDraw
import sys
sys.path.insert(1,r"C:\Users\PC\Documents\GitHub\EMPIS LAB")

# Define tile colors
tile_colors = {
    "solid": (0, 0, 255),  # Blue
    "breakable": (200, 80, 70),  # Yellow
    "passable": (0, 255, 0),  # Green
    "empty": (173, 216, 230),  # ligth blue
    "question block": (255, 165, 0),  # Orange
    "full question block": (255, 69, 0),  # Red-Orange
    "damaging": (100,0,0),  # Red
    "hazard": (128, 0, 128),  # Purple
    "moving": (255, 0, 255),  # Magenta
    "top-left pipe": (0, 200, 100),  # D. Green
    "top-right pipe": (0, 200, 100),  # D. Green
    "left pipe": (0, 230, 0),  # Green
    "right pipe": (0, 230, 0),  # Green
    "collectable": (255, 255, 0),  # Gold
    "ground": (160, 50, 40),  # brick
    "cannon": (75,83,32)  # Maroon
}



def create_mario_level(csv_file_path, output_image_path):
    with open(csv_file_path, 'r') as file:
        level_data = [list(line) for line in file]
        
    for y, row in enumerate(level_data):
        for x, tile in enumerate(row):
            if tile == ',':
                row.pop(x)

    tile_size = 30  # Size of each tile in pixels
    image_width = len(level_data[0]) * tile_size
    image_height = len(level_data) * tile_size

    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for y, row in enumerate(level_data):
        for x, tile in enumerate(row):
            if tile in mario_tiles["tiles"]:
                tile_type = mario_tiles["tiles"][tile]
                color = tile_colors[tile_type[1]]
                draw.rectangle(
                    [(x * tile_size, y * tile_size),
                     ((x + 1) * tile_size, (y + 1) * tile_size)],
                    fill= color
                )

    image.save(output_image_path)

# Example usage
mario_tiles = {
    "tiles": {
        "X": ["solid", "ground"],
        "S": ["solid", "breakable"],
        "-": ["passable", "empty"],
        "?": ["solid", "question block", "full question block"],
        "Q": ["solid", "question block", "empty question block"],
        "E": ["enemy", "damaging", "hazard", "moving"],
        "<": ["solid", "top-left pipe", "pipe"],
        ">": ["solid", "top-right pipe", "pipe"],
        "[": ["solid", "left pipe", "pipe"],
        "]": ["solid", "right pipe", "pipe"],
        "o": ["coin", "collectable", "passable"],
        "B": ["Cannon top", "cannon", "solid", "hazard"],
        "b": ["Cannon bottom", "cannon", "solid"]
    }
}


csv_file_path = r'../Super_Mario_Brothers_Maps/final_levels/1-1.txt'.format('1-2')
output_image_path = r'../Super_Mario_Brothers_Maps/final_levels/1-1.png'.format('1-2')
create_mario_level(csv_file_path, output_image_path)

example_code = ['1-1','1-2']

csv_file_path = r'../Super_Mario_Brothers_Maps/final_levels/Performance/{}_opt=0.txt'.format(example_code)
output_image_path = r'../Super_Mario_Brothers_Maps/final_levels/Performance/{}_opt=0.png'.format(example_code)

create_mario_level(csv_file_path, output_image_path)

csv_file_path = r'../Super_Mario_Brothers_Maps/final_levels/Performance/{}_opt=0.1.txt'.format(example_code)
output_image_path = r'../Super_Mario_Brothers_Maps/final_levels/Performance/{}_opt=0.1.png'.format(example_code)

create_mario_level(csv_file_path, output_image_path)

csv_file_path = r'../Super_Mario_Brothers_Maps/final_levels/Performance/{}_opt=1.txt'.format(example_code)
output_image_path = r'../Super_Mario_Brothers_Maps/final_levels/Performance/{}_opt=1.png'.format(example_code)

create_mario_level(csv_file_path, output_image_path)



csv_file_path = r'../Super_Mario_Brothers_Maps/final_levels/Landings_Score/{}_opt=0.txt'.format(example_code)
output_image_path = r'../Super_Mario_Brothers_Maps/final_levels/Landings_Score/{}_opt=0.png'.format(example_code)

create_mario_level(csv_file_path, output_image_path)

csv_file_path = r'../Super_Mario_Brothers_Maps/final_levels/Landings_Score/{}_opt=1.txt'.format(example_code)
output_image_path = r'../Super_Mario_Brothers_Maps/final_levels/Landings_Score/{}_opt=1.png'.format(example_code)

create_mario_level(csv_file_path, output_image_path)



