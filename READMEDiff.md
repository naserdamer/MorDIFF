To morph two image with diffusion model, two steps are needed:

0. Place diffusion model folder in same directory as files
- morph_two_images.py
- morph_two_images.sh
- folder to diffusion model (example: diffae-master)
- folder to both images (example: test)

1. In ./diffae-master align both images with 
python ./diffae-master/align.py --input_imgs_path {path to image folder} --output_imgs_path {path to image folder}

2. Morph two images using
python morph_two_images.py --img1 {path to img1 aligned} --img2 {path to image2 aligned} --output {output path where to store morphed image}

This generates two images, 
1. morphed image, 
2. comparison image:  img1, morphed, img2

In morph_two_images.py change sys.path.append(...) to directory of diffusion model (example: ./diffae-master)
Pay attention to place the corresponding model weights in the folder according to torch.load(...) in line 39 in morph_two_images.py