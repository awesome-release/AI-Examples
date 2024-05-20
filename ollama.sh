#!/bin/bash
# Ollama


echo "===== Downloading model from S3..."
cp -R /bucket/axolotol-out-2024-05-08-16-50-08 /models/

echo "===== Converting the model to .bin..."
python llm/llama.cpp/convert.py /models/axolotol-out-2024-05-08-16-50-08 --outtype f16 --outfile /models/converted.bin

echo "===== Quantizing model..."
llm/llama.cpp/quantize /models/converted.bin /models/quantized.bin q4_0
