#!/bin/bash

# This script is used to install git, docker, and docker-compose on an EC2 instance.
# By including it as user data in the launch configuration of the instance, it will be 
# executed when the instance is launched.

# Update package manager and install Docker and Git
sudo yum update -y
sudo yum install docker git -y

# Start Docker service
sudo systemctl start docker

# Add ec2-user to the docker group
sudo usermod -a -G docker ec2-user

# Enable Docker service to start on boot
sudo systemctl enable docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
