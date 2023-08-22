import os
from PIL import Image
from tqdm import tqdm

imgs = os.listdir('./data')
for img_name in tqdm(imgs, total=len(imgs)):
    img = Image.open(os.path.join('./data', img_name)).convert('RGB')
    name, ext = img_name.split('.')
    img.save(f'./data/{name}.png')
    if ext != 'png':
        os.remove(os.path.join('./data', img_name))
