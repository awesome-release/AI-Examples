FROM nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.08.03

COPY scripts/* /bin/
COPY cpu.merge.conf.yaml /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/conf/merge_lora_weights.yaml
COPY scripts/merge.py /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py