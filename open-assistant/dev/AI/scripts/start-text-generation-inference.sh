#model=bigscience/bloom-560m
model=OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
num_shard=1
volume=$PWD/../../inference-data # share a volume with the Docker container to avoid downloading weights every run
name="text-generation-inference"
docker run --rm --name $name --gpus all --shm-size 1g -p 8081:80 \
    -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id $model --num-shard $num_shard \
    --disable-custom-kernels