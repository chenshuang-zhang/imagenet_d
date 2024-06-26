
# CUDA_VISIBLE_DEVICES='0,1,2,3,4' python -m llava.serve.eval_imagenet_d \
#     --model-path ./pretrained_weights/llava-v1.6-34b/ \
#     --experiment_name 'texture' \

# CUDA_VISIBLE_DEVICES='0,1,2,3,4' python -m llava.serve.eval_imagenet_d \
#     --model-path ./pretrained_weights/llava-v1.6-34b/ \
#     --experiment_name 'material' \

# CUDA_VISIBLE_DEVICES='0,1,2,3,4' python -m llava.serve.eval_imagenet_d \
#     --model-path ./pretrained_weights/llava-v1.6-34b/ \
#     --experiment_name 'background' \

# CUDA_VISIBLE_DEVICES='0,1,2,3,4' python -m llava.serve.eval_all_dataset \
#     --model-path ./pretrained_weights/llava-v1.6-34b/ \
#     --experiment_name 'imagenet' \

# CUDA_VISIBLE_DEVICES='0,1,2,3,4' python -m llava.serve.eval_all_dataset \
#     --model-path ./pretrained_weights/llava-v1.6-34b/ \
#     --experiment_name 'imagenet9' \

# CUDA_VISIBLE_DEVICES='0,1,2,3,4' python -m llava.serve.eval_all_dataset \
#     --model-path ./pretrained_weights/llava-v1.6-34b/ \
#     --experiment_name 'stylized_imagenet' \

CUDA_VISIBLE_DEVICES='0,1,2,3' python -m llava.serve.eval_all_dataset \
    --model-path ./pretrained_weights/llava-v1.6-34b/ \
    --experiment_name 'objectnet' \
