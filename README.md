# AI-Examples

## Basic Flow

### NeMo-Training

- Convert HF format to NeMo format
- Run fine tuning
- Merge fine tuned NeMo model into Pre-trained NeMo model
- Convert .nemo back to HF

### TensorRT-LLM

- Build Triton engine from merged HF model

### TritonServer

- Serve Triton engine via Triton Server

## Fine Tuning

### Configure cluster
```
KUBECONFIG=./config-release-ry6clz\ \(3\).yaml kubectl apply -f configmap.yml

cat ~/Downloads/configmap.yml                                                 ──(Wed,Mar13)─┘

apiVersion: v1
kind: ConfigMap
metadata:
  name: kubelet-config-1.21
  namespace: kube-system
data:
  kubelet: |
    kubeletArguments:
      runtimeRequestTimeout: "20m"
```

### Acquire models

From a shell within the Nemo app, copy the models from s3 to /models

These are pre-downloaded and formatted for Nemo consumption, and also include pre-processed fine tuning datasets. Directions on how this was done are in the SFT and PEFT tutorial links below.

```
aws s3 sync s3://release-ry6clz-static-builds/ai-models-tmp/ /models/
```

Currently contains multiple models for testing. You can also only grab what you need to save time/bandwidth.

#### s3 Bucket

- Llama-2-7b-hf/ - https://huggingface.co/meta-llama/Llama-2-7b-hf
- databricks-dolly-15k/ - Example fine tuning data set. Used with https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2sft.html
- llama-recipes/ - https://github.com/facebookresearch/llama-recipes/
- llama.cpp/ - https://github.com/ggerganov/llama.cpp
- nemo_experiments/ - Fine tuning outputs
- pubmedqa/ - Example fine tuning data set. Used with https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2peft.html
- results/
- trt_engines-fp16-4-gpu/ - Engine built from Llama-2-7b-hf following https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/
- ReleaseLlama.ipynb - Example jupyter notebook
- llama-2-7b-hf.nemo - Nemo formatted model
- release.pdf - Release docs for RAG

### Convert HF to .nemo

python scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=/workspace/Llama-2-7b-hf/ --out-file=/workspace/llama2-7b.nemo

### Run Fine Tuning

These articles are the latest tutorials on running training.
[SFT Tutorial](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2sft.html)

## SFT

Current SFT command:

```

export CONCAT_SAMPLING_PROBS="[1]"
export TP_SIZE=8
export PP_SIZE=1
export CONCAT_SAMPLING_PROBS="[1.0]"
export MODEL="/models/llama-2-7b-hf.nemo"
export TRAIN_DS="[/models/databricks-dolly-15k/training.jsonl]"
export VALID_DS="[/models/databricks-dolly-15k/validation.jsonl]"
export TEST_DS="[/models/databricks-dolly-15k/test.jsonl]"
export VALID_NAMES="[databricks-dolly-15k]"

torchrun --nproc_per_node=8 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
   trainer.precision=bf16 \
   trainer.devices=8 \
   trainer.num_nodes=1 \
   trainer.val_check_interval=0.1 \
   trainer.max_steps=50 \
   model.restore_from_path=${MODEL} \
   model.micro_batch_size=1 \
   model.global_batch_size=128 \
   model.tensor_model_parallel_size=${TP_SIZE} \
   model.pipeline_model_parallel_size=${PP_SIZE} \
   model.megatron_amp_O2=True \
   model.sequence_parallel=True \
   model.activations_checkpoint_granularity=selective \
   model.activations_checkpoint_method=uniform \
   model.optim.name=distributed_fused_adam \
   model.optim.lr=5e-6 \
   model.answer_only_loss=True \
   model.data.train_ds.file_names=${TRAIN_DS} \
   model.data.validation_ds.file_names=${VALID_DS} \
   model.data.test_ds.file_names=${TEST_DS} \
   model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
   model.data.train_ds.max_seq_length=2048 \
   model.data.validation_ds.max_seq_length=2048 \
   model.data.train_ds.micro_batch_size=1 \
   model.data.train_ds.global_batch_size=128 \
   model.data.validation_ds.micro_batch_size=1 \
   model.data.validation_ds.global_batch_size=128 \
   model.data.test_ds.micro_batch_size=1 \
   model.data.test_ds.global_batch_size=256 \
   model.data.train_ds.num_workers=0 \
   model.data.validation_ds.num_workers=0 \
   model.data.test_ds.num_workers=0 \
   model.data.validation_ds.metric.name=loss \
   model.data.test_ds.metric.name=loss \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=/models/results \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss \
   exp_manager.checkpoint_callback_params.save_best_model=False \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   ++cluster_type=BCP
```

The output of this command lives at `/models/results/checkpoints/megatron_gpt_sft.nemo` in the s3 bucket.

```
s3://release-ry6clz-static-builds/ai-models-tmp/results/checkpoints/megatron_gpt_sft.nemo
```

## PEFT

[PEFT Tutorial](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2peft.html)

Current PEFT command:

```
export CONCAT_SAMPLING_PROBS="[1]"
export TP_SIZE=4
export PP_SIZE=1
export CONCAT_SAMPLING_PROBS="[1.0]"
export MODEL="/tmp/tinyllama-1.1b-chat-v1.nemo"
export TRAIN_DS="[/bucket/ai-models-tmp/pubmedqa/pubmedqa_train.jsonl]"
export VALID_DS="[/bucket/ai-models-tmp/pubmedqa/pubmedqa_val.jsonl]"
export TEST_DS="[/bucket/ai-models-tmp/pubmedqa/pubmedqa_test.jsonl]"
export TEST_NAMES="[pubmedqa]"
export SCHEME="lora"
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
torchrun --nproc_per_node=4 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
    trainer.devices=4 \
    trainer.num_nodes=1 \
    trainer.precision=bf16 \
    trainer.val_check_interval=20 \
    trainer.max_steps=50 \
    model.megatron_amp_O2=False \
    ++model.mcore_gpt=True \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.micro_batch_size=1 \
    model.global_batch_size=4 \
    model.restore_from_path=${MODEL} \
    model.data.train_ds.num_workers=0 \
    model.data.validation_ds.num_workers=0 \
    model.data.train_ds.file_names=${TRAIN_DS} \
    model.data.train_ds.concat_sampling_probabilities=[1.0] \
    model.data.validation_ds.file_names=${VALID_DS} \
    model.peft.peft_scheme=${SCHEME}
```

To run fine tuning, spin up an env for the [Nemo app in release](https://app.release.com/admin/apps/8118/environments).

### Merge

Run `merge.py` from this repo with `cpu.merge.conf.yaml` settings on `nvcr.io/nvidia/nemo:23.10`

#### Extract .nemo model and fix it.
cd /models
mkdir NemoModelExtracted
cd NemoModelExtracted
tar -xvf ../model_to_extract.nemo

edit the model_config.yaml file and make global_batch_size a large power of 2, say 256
instead of the nemo file in gpt_model_file

cp THISREPO/cpu.merge.conf.yaml /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/conf/model_config.yml
cp THISREPO/scripts/merge.py /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py
cd /opt/NeMo/
python scripts/nlp_language_modeling/merge_lora_weights/merge.py

## RAG

For Retrieval Augmented Generation, reference is here https://github.com/NVIDIA/GenerativeAIExamples/tree/main/RetrievalAugmentedGeneration/

### Cleanup

When you're done, if you want to save the new model state back to S3, reverse the `aws s3 sync` command from above.

```
aws s3 sync /models/ s3://release-ry6clz-static-builds/ai-models-tmp/
```

Also be sure to delete your Env when done with fine tuning and/or testing. They are costly.

# TensorRT-LLM (Training Server)

Building tensorrt-llm so we can train our models.
Spin up a g5 instance.

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y --reinstall make docker-ce git-lfs
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
make -C docker release_build
```

https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/

# Other Release AI Apps

These are other Release apps you can use to interact with the models.

The [Jupyter notebook](https://app.release.com/admin/apps/8117/environments) is good for testing. You can run things like https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/HelloLlamaLocal.ipynb or ReleaseLlama.ipynb in this repository.

The [NeMo launcher](https://app.release.com/admin/apps/8118/environments) app is for runing SFT and PEFT jobs.

The [TensorRT-LLM](https://app.release.com/admin/apps/8169/environments) app is where you can compile the models into engines to be used with the Triton inference server.

The [Triton inference](https://app.release.com/admin/apps/8107/environments) server will provide an endpoint for the Chatbot. Currently need to get TensorRT-LLM compiled in a docker image to be able to use it. Ref: https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md

## Tips

Monitor GPU status with

```
while true; do nvidia-smi && sleep 5; done
```

## Compile the model

Docs: https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0/examples/llama

From within the TensorRT-LLM docker image

```
663072083902.dkr.ecr.us-west-2.amazonaws.com/awesome-release/tensorrt-llm/tensorrtllm:latest
```

# Extract model
```
cp /bucket/ai-models-tmp/tiny_merged_model.nemo /models/
cd /models
tar -xvf ../tiny_merged_model.nemo -C tiny_merged_model/
```

# Build engines

```
cd /app/tensorrt_llm
python examples/llama/build.py --model_dir /models/Llama-2-7b-hf.20240307015747.tuned.merged-hf/ \
  --dtype float16 \
  --remove_input_padding \
  --use_gpt_attention_plugin float16 \
  --enable_context_fmha \
  --use_gemm_plugin float16 \
  --output_dir /models/Llama2-7b-tuned-merged_engine-fp16-4-gpu/ \
  --world_size 4 \
  --tp_size 2 \
  --pp_size 2 \
  --use_inflight_batching \
  --paged_kv_cache
```

# Export from .nemo to TensorRT-LLM engine

python scripts/export/export_to_trt.py --nemo_checkpoint /models/tiny_merged_model.nemo --model_type="llama" --model_repository /models/trt_tiny-fp16-4gpu

# Hmmm

Test:
python scripts/export/export_to_trt.py --nemo_checkpoint /models/GPT-2B-001_bf16_tp1.nemo --num_gpus 4 --model_type="gptnext" --model_repository /models/tmp_model_repository/


## In Inference:
python scripts/export/export_to_trt.py --num_gpus 1 --nemo_checkpoint /models/TinyLlama-1.1B-Chat-v1.0.nemo --model_type="llama" --model_repository /models/trt_tiny_untuned_1gpu

python scripts/export/export_to_trt.py --num_gpus 1 --nemo_checkpoint /models/tiny_merged_model.nemo --model_type="llama" --model_repository /models/trt_tiny_merged_1gpu

mpirun -n 4 --allow-run-as-root \
python scripts/export/export_to_trt.py --num_gpus 4 --nemo_checkpoint /models/tiny_merged_model.nemo --model_type="llama" --model_repository /models/trt_tiny_merged_4gpu

## In TensortRT-LLM:
mpirun -n 4 --allow-run-as-root \
python3 examples/llama/run.py\
  --engine_dir=/models/trt_tiny-fp16-4gpu \
  --max_output_len 100 \
  --tokenizer_dir /models/TinyLlama-1.1B-Chat-v1.0/ \
  --input_text "What is ReleaseHub.com"
  
# Use existing engine
```

cp /bucket/ai-models-tmp/tiny_merged_model.nemo /models/
cp -R /bucket/ai-models-tmp/trt_tiny-fp16-4gpu /models/


mpirun -n 4 --allow-run-as-root \
python3 examples/llama/run.py\
  --engine_dir=/models/trt_engines-fp16-4-gpu \
  --max_output_len 100 \
  --tokenizer_dir /models/Llama-2-7b-hf/ \
  --input_text "What is ReleaseHub.com"
```

### Test the model via TensorRT-LLM

#### 1 gpu
tar -xvzf tiny_merged_model.nemo -C tiny_merged_model/
modify /models/trt_tiny_1gpu/config.json
add
```
  "pipeline_parallel": 1,
  "quant_mode": false,
```

##### Convert merged .nemo back to HF from `nemofw-training`

cp /bucket/ai-models-tmp/tiny_merged_model.nemo /models/
cp -r /bucket/ai-models-tmp/TinyLlama-1.1B-Chat-v1.0 /models/
python scripts/nlp_language_modeling/convert_nemo_llama_to_hf.py --in-file /models/tiny_merged_model.nemo --out-file /models/tiny_merged_model_hf.bin --hf-in-path /models/TinyLlama-1.1B-Chat-v1.0/ --hf-out-path /models/tiny_merged_model_hf/

cp 

python scripts/nlp_language_modeling/convert_nemo_llama_to_hf.py --in-file /models/Llama-2-7b-hf.20240307015747.nemo.tuned.nemo --out-file /models/Llama-2-7b-hf.20240307015747.bin --hf-in-path /models/Llama-2-7b-hf/ --hf-out-path /models/Llama-2-7b-hf.20240307015747.tuned.merged-hf/ --cpu-only

##### Test against tuned+merged engine

python3 examples/llama/run.py\
  --engine_dir=/models/trt_tiny_merged_1gpu \
  --max_output_len 100 \
  --tokenizer_dir /models/TinyLlama-1.1B-Chat-v1.0/ \
  --input_text "Is gastric cancer different in Korea and the United States"

##### Test against untuned engine

python3 examples/llama/run.py\
  --engine_dir=/models/trt_tiny_untuned_1gpu \
  --max_output_len 100 \
  --tokenizer_dir /models/TinyLlama-1.1B-Chat-v1.0/ \
  --input_text "Is gastric cancer different in Korea and the United States"

#### 4 GPU: 
untar .nemo to tiny_merged_model/
modify tiny_merged_model/config.json
add
```
  "pipeline_parallel": 4,
  "quant_mode": false,
```

mpirun -n 4 --allow-run-as-root \
python3 examples/llama/run.py\
  --engine_dir=/models/trt_tiny-fp16-4gpu \
  --max_output_len 100 \
  --tokenizer_dir /models/tiny_merged_model/ \
  --input_text "What is ReleaseHub.com"
```

mpirun -n 4 --allow-run-as-root \
python3 /app/tensorrt_llm/examples/llama/run.py\
  --engine_dir=/models/trt_tiny-fp16-4gpu \
  --max_output_len 100 \
  --tokenizer_dir /models/tiny_merged_model.nemo \
  --input_text "What is ReleaseHub.com"

### Run the model via Triton

```
cd tensorrtllm_backend

# Set up the model repository
cp ../TensorRT-LLM/trt_engines-fp16-4-gpu/*   all_models/inflight_batcher_llm/tensorrt_llm/1/
python3 tools/fill_template.py --in_place \
      all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
      decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,\
max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,\
max_num_sequences:4
python3 tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:meta-llama/Llama-2-7b-chat-hf
python3 tools/fill_template.py --in_place \
    all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_type:llama,tokenizer_dir:meta-llama/Llama-2-7b-chat-hf

docker run -it --rm --gpus all --network host --shm-size=1g \
-v $(pwd)/all_models:/all_models \
-v $(pwd)/scripts:/opt/scripts \
nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3

huggingface-cli login

pip install sentencepiece protobuf

python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 4
```

## Query the model

```
curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
"text_input": "What is ReleaseHub.com",
"parameters": {
"max_tokens": 100,
"bad_words":[""],
"stop_words":[""]
}
}'
```

wget --method POST --body-data '{
"text_input": "What is ReleaseHub.com",
"parameters": {
"max_tokens": 100,
"bad_words":[""],
"stop_words":[""]
}
}' --header='Content-Type: application/json' -O - localhost:8000/v2/models/Llama-2-7b-hf-20240305-1.tuned/generate

## Convert .nemo to .onnx

```
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/core/export.html

## Start Triton with ONNX model

```
tritonserver --model-repository=/models
```

## Convert .nemo to HF

```
python scripts/nlp_language_modeling/convert_nemo_llama_to_hf.py \
--in-file /models/tiny_merged_model_hf.20240404155140.nemo.tuned.nemo \
--out-file /models/tiny_merged_model_hf.20240404155140.bin \
--hf-in-path /models/TinyLlama-1.1B-Chat-v1.0/ \
--hf-out-path /models/tiny-releasetuned-20240404-hf/
```
Copy the `tokenizer*` files and `special_tokens_map.json` files from the hf-in-path to hf-out-path.

# Ray

## Ray-LLM

Install operator
```
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0
```

Apply k8s manifest
```
kubectl apply -f ray/ray-cluster.rayllm-eks.yaml
```

- Can use ray/multi.yaml as a serve_config/
- Put ray/models/* into models/

