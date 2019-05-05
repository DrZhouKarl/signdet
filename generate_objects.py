import Augmentor

p = Augmentor.Pipeline("igvc_objs")
p.random_distortion(probability=1, grid_width=8, grid_height = 8, magnitude=3)
p.gaussian_distortion(1, 
