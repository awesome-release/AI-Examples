FROM 663072083902.dkr.ecr.us-west-2.amazonaws.com/awesome-release/tensorrt-llm/tensorrtllm:latest

COPY scripts/peft_tuning /bin/peft_tuning

CMD ["/bin/peft_tuning"]
