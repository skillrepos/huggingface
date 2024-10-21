HUGGINGFACE_TOKEN=$(cat /home/vscode/.cache/huggingface/token)
HUGGINGFACE_USER=$(huggingface-cli whoami)
git remote set-url origin https://$HUGGINGFACE_USER:$HUGGINGFACE_TOKEN@huggingface.co/$HUGGINGFACE_USER/$1
