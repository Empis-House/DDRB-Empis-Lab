import cv2

import numpy as np
import matplotlib.pyplot as plt

import numba as nb


##variable to get the specific level

level_name = "1-1"
opt = "1"

##function to get the level image

def get_level_image(level_name):
    level_image = cv2.imread(f"../Super_Mario_Brothers_Maps/Original/mario-{level_name}.png")
    return level_image


test_level = get_level_image(level_name)



##function to get the image separated into tiles as numpy array

def get_tileset(level, tile_size):
    height, width, _ = level.shape
    
    tileset = np.array([level[y:y+tile_size, x:x+tile_size] for y in range(0, height, tile_size) for x in range(0, width, tile_size)])
    
    ## set as a 2d grid
    tileset = tileset.reshape(height//tile_size,width//tile_size, tile_size, tile_size, 3)
    
    return tileset

tile_size = 16

tileset = get_tileset(test_level, tile_size)

## funciton to get text version of the level as numpy array
def get_text_file(level_path, separator = ""):
    text_array = []
    #read the text file
    with open(level_path) as f:
        level_text = f.readlines()[1:]
        for line in level_text:
            text_array.append(list(line.replace(separator,"").strip()))
    return np.array(text_array)

test_level_text = get_text_file(f"../Super_Mario_Brothers_Maps/Processed/mario-1-1.txt")


#--------------------------unnecessary code but i keep it just in case--------------------------------
# def phash(image):
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
#         h=cv2.img_hash.pHash(img) # 8-byte hash
#         pH=int.from_bytes(h.tobytes(), byteorder='big', signed=False)
#         return pH
    
    

# def get_tileset_hash(tileset):
#     tileset_hash = np.array([[phash(tile) for tile in row] for row in tileset])
#     return tileset_hash

# # @nb.njit('void(int_[:,::1], int_[::1])', parallel=True)
# def clean_background(a, b):
#     n, m = a.shape
#     for i in range(n):
#         for j in range(m):
#             if a[i,j] == '-':
#                 b[i,j] = 0.0
#     return b

# tileset_hash = get_tileset_hash(tileset)
# tileset_hash_clean = clean_background(test_level_text, tileset_hash)
#---------------------------------------------------------------------------------------------

text_to_tile = {test_level_text[i,j]:tileset[i,j] for i in range(tileset.shape[0]) for j in range(tileset.shape[1])}

print(text_to_tile["-"].shape)

def rebuild_image(level_text, text_to_tile):
    height, width = level_text.shape
    image = np.zeros((height*tile_size, width*tile_size, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = text_to_tile[level_text[i,j]]
    return image

# print(test_level.shape)

new_level_text = get_text_file("../Super_Mario_Brothers_Maps/final_levels/Performance/1-1_opt=0.1.txt", separator=",")
rebuilt_image = rebuild_image(new_level_text, text_to_tile)
cv2.imshow("Tile test", rebuilt_image)    
cv2.waitKey(0)
cv2.destroyAllWindows()