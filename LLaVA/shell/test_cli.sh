IMAGEPATH='../data/background/spatula/mPupNRJPYA2DPFGht9P8XZ.png'

CUDA_VISIBLE_DEVICES='1,2' python -m llava.serve.cli \
    --model-path ./pretrained_weights/llava-v1.5-13b/ \
    --image-file $IMAGEPATH 