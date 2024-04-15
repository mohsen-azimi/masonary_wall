import subprocess
import socket

def git_add_commit_push():
    subprocess.run(["git", "add", "*.py"])
    subprocess.run(["git", "add", "*.md"])
    # subprocess.run(["git", "add", "*.txt"])
    subprocess.run(["git", "commit", "-m", "updated"])
    subprocess.run(["git", "push", '-f'])
    # git gc --prune=30.days.ago
    # subprocess.run(["git", "gc", "--prune=30.days.ago"])

if __name__ == '__main__':
    git_add_commit_push()

