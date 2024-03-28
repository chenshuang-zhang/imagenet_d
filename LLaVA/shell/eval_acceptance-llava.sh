
CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_imagenet_d \
    --model-path ./pretrained_weights/llava-v1.6-vicuna-13b/ \
    --experiment_name 'background' \

CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_imagenet_d \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'texture' \

CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_imagenet_d \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'material' \

CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_imagenet_d \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'background' \


CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_all_dataset \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'imagenet' \

CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_all_dataset \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'imagenet9' \

CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_all_dataset \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'stylized_imagenet' \

CUDA_VISIBLE_DEVICES='0,1' python -m llava.serve.eval_all_dataset \
    --model-path /media/philipp/ssd1_2tb/zcs_projects/homework/experiment/experiment2/benchmark/LLaVA/pretrained_weights/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3/ \
    --experiment_name 'objectnet' \