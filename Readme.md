# ROS GCN

## Setup
If you opened with Gitpod, you can skip the Setup section!

First build the docker images:
```bash
docker build -t smile_ros -f ros.Dockerfile .
docker build -t rosboard -f rosboard.Dockerfile .
```

Then pull and run Dozzle, a Docker image that's helpful for viewing logs:
```bash
docker pull amir20/dozzle:latest
docker run --name dozzle -d --volume=/var/run/docker.sock:/var/run/docker.sock -p 8888:8080 amir20/dozzle:latest
```

## Running
```bash
docker-compose up -d
```
