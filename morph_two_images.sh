IMG_FOLDER_IN="/home/psiebke/diffusion/test/"
IMG_FOLDER_ALIGN="/home/psiebke/diffusion/test/aligned/"

python ./diffae-master/align.py --input_imgs_path $IMG_FOLDER_IN --output_imgs_path $IMG_FOLDER_ALIGN

IMG1="/home/psiebke/diffusion/test/aligned/001_03.png"
IMG2="/home/psiebke/diffusion/test/aligned/019_03.png"

OUTPUT="/home/psiebke/diffusion/test/morphed/"

python morph_two_images.py --img1 $IMG1 --img2 $IMG2 --output $OUTPUT