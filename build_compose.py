import argparse
import pathlib
import yaml

thisdir = pathlib.Path(__file__).resolve().parent

ROS_IMAGE = "smile_ros"

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-robots", 
        type=int, required=True,
        help="Number of robots in system"
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    dockerfile = {
        "version": "3",
        "services": {}
    }

    build_commands = ["colcon build", ". ./install/setup.sh"]
    all_nodes_str = ",".join([f"node_{j}" for j in range(1, args.num_robots+1)])

    scheduler_commands = [*build_commands, "ros2 run gcnsched_demo scheduler"]
    dockerfile["services"]["scheduler"] = {
        "image": ROS_IMAGE,
        "volumes": [
            f'{thisdir.joinpath(".workspaces/scheduler")}:/workspace',
            f'{thisdir.joinpath("src")}:/workspace/src:ro'
        ],
        "environment": {
            "PYTHONUNBUFFERED": 1,
            "ALL_NODES": all_nodes_str
        },
        "command": f'bash -c "{" && ".join(scheduler_commands)}"'
    }
    
    rosboard_commands = [*build_commands, "/rosboard/run"]
    dockerfile["services"]["rosboard"] = {
        "image": "rosboard",
        "volumes": [
            f'{thisdir.joinpath(".workspaces/rosboard")}:/workspace',
            f'{thisdir.joinpath("src")}:/workspace/src:ro'
        ],
        "environment": {
            "PYTHONUNBUFFERED": 1
        },
        "ports": ["9999:8888"],
        "command": f'bash -c "{" && ".join(rosboard_commands)}"'
    }

    for i in range(1, args.num_robots+1):
        
        executor_commands = [*build_commands, "ros2 run gcnsched_demo executor"]
        dockerfile["services"][f"executor_{i}"] = {
            "image": ROS_IMAGE,
            "volumes": [
                f'{thisdir.joinpath(f".workspaces/executor_{i}")}:/workspace',
                f'{thisdir.joinpath("src")}:/workspace/src:ro'
            ],
            "environment": {
                "PYTHONUNBUFFERED": 1,
                "ALL_NODES": all_nodes_str,
                "NODE_NAME": f"node_{i}"
            },
            "command": f'bash -c "{" && ".join(executor_commands)}"'
        }


        bandwidth_commands = [*build_commands, "ros2 run gcnsched_demo bandwidth"]
        dockerfile["services"][f"bandwidth_{i}"] = {
            "image": ROS_IMAGE,
            "volumes": [
                f'{thisdir.joinpath(f".workspaces/bandwidth_{i}")}:/workspace',
                f'{thisdir.joinpath("src")}:/workspace/src:ro'
            ],
            "environment": {
                "PYTHONUNBUFFERED": 1,
                "ALL_NODES": all_nodes_str,
                "NODE_NAME": f"node_{i}"
            },
            "command": f'bash -c "{" && ".join(bandwidth_commands)}"'
        }

    thisdir.joinpath("docker-compose.yml").write_text(yaml.dump(dockerfile))
    

if __name__ == "__main__":
    main()