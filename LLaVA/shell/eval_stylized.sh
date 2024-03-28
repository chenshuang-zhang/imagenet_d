CUDA_VISIBLE_DEVICES='2,3' python -m llava.serve.eval_all_dataset \
    --model-path ./pretrained_weights/llava-v1.5-13b/ \
    --experiment_name 'imagenet9' \

CUDA_VISIBLE_DEVICES='2,3' python -m llava.serve.eval_all_dataset \
    --model-path ./pretrained_weights/llava-v1.5-13b/ \
    --experiment_name 'stylized_imagenet' \


CUDA_VISIBLE_DEVICES='2,3' python -m llava.serve.eval_all_dataset \
    --model-path ./pretrained_weights/llava-v1.6-vicuna-13b/ \
    --experiment_name 'imagenet9' \

CUDA_VISIBLE_DEVICES='2,3' python -m llava.serve.eval_all_dataset \
    --model-path ./pretrained_weights/llava-v1.6-vicuna-13b/ \
    --experiment_name 'stylized_imagenet' \