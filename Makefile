build:
	docker build -t weightless-neural-networks .
    docker tag weightless-neural-networks viniciusdsmello/weightless-neural-networks:latest

push:
	docker push viniciusdsmello/weightless-neural-networks:latest

singularity-build:
	singularity pull docker://viniciusdsmello/weightless-neural-networks:latest