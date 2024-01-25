IMAGE_NAME = "awesome-release/peft_tuning"
IMAGE_TAG = "latest"

build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
