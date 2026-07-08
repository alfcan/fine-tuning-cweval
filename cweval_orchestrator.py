import sys
import os
import shutil
import datetime
from pathlib import Path

# Add CWEval directory to python path so we can import sandbox
cweval_path = Path(__file__).parent / "CWEval"
sys.path.insert(0, str(cweval_path))

# pyrefly: ignore [missing-import]
from cweval.sandbox import Container

def run_evaluation_in_docker(eval_path, num_proc=8):
    # Configure Docker Desktop socket on macOS if present
    mac_docker_sock = os.path.expanduser("~/.docker/run/docker.sock")
    if os.path.exists(mac_docker_sock):
        os.environ["DOCKER_HOST"] = f"unix://{mac_docker_sock}"

    # Absolute path on host
    eval_path_abs = Path(eval_path).resolve()
    
    # 1. Start the container
    timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    container_name = f'cweval_{timestamp}'
    print(f"Starting Docker container (co1lin/cweval) named {container_name}...")
    container = Container(
        image='co1lin/cweval',
        name=container_name
    )
    
    try:
        # 2. Prepare paths in container
        repo_path_in_docker = '/home/ubuntu/CWEval'
        evals_path_in_docker = f"{repo_path_in_docker}/evals"
        eval_path_in_docker = f"{evals_path_in_docker}/{eval_path_abs.name}"
        
        # Clean folder in container
        container.exec_cmd(f'bash -c "mkdir -p {evals_path_in_docker} && rm -rf {eval_path_in_docker}"')
        
        # 3. Copy host eval_path to container
        print(f"Copying {eval_path_abs} to container at {eval_path_in_docker}...")
        container.copy_to(str(eval_path_abs), eval_path_in_docker)
        
        # 4. Execute the pipeline inside the container
        # Since we are running inside docker, we pass --docker False
        cmd = f'''bash -c "
        source /home/ubuntu/miniforge3/bin/activate;
        cd {repo_path_in_docker};
        source .env;
        python cweval/evaluate.py pipeline --eval_path {eval_path_in_docker} --num_proc {num_proc} --docker False;
        "'''
        
        print("Running evaluation pipeline inside Docker container...")
        exit_code, stdout, stderr = container.exec_cmd(cmd)
        
        # Print logs/output
        if stdout:
            print("--- CONTAINER STDOUT ---")
            print(stdout)
        if stderr:
            print("--- CONTAINER STDERR ---", file=sys.stderr)
            print(stderr, file=sys.stderr)
            
        if exit_code != 0:
            raise RuntimeError(f"Evaluation pipeline failed inside Docker with exit code {exit_code}")
            
        # 5. Copy the results back to the host
        print(f"Copying results back to {eval_path_abs}...")
        # Clear the host folder first to avoid conflict and clean up old results
        if eval_path_abs.exists():
            shutil.rmtree(eval_path_abs)
        container.copy_from(eval_path_in_docker, str(eval_path_abs))
        
    finally:
        print("Stopping and cleaning up Docker container...")
        # container will be killed and removed on destruction
        del container
