from utils import draw_at_random          


root_path = 'cat2000'
x = 'train'


imgs_path = lambda x: f'{root_path}/{x}/Stimuli/'

imgs_path = imgs_path(x)


draw_at_random(imgs_path)



