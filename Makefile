CLUSTER_NAME := fti-cluster
VOLUME_PATH := /Users/r0d3r1ch25/opt/fti-pipeline/data/notebooks


cluster-up:
	k3d cluster create $(CLUSTER_NAME) \
		--servers 1 --agents 2 \
		-v "$(VOLUME_PATH):/home/jovyan/work@all"
	kubectl cluster-info

cluster-down:
	k3d cluster delete $(CLUSTER_NAME)

validate:
	kustomize build base | kubectl apply --dry-run=client -f -

deploy:
	kubectl apply -k base

clean:
	kubectl delete -k base


.PHONY: cluster-up cluster-down validate deploy clean
