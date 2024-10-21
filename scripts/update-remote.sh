HUGGINGFACE_TOKEN=$(cat /home/vscode/.cache/huggingface/token)
git remote set-url origin https://$1:$HUGGINGFACE_TOKEN@huggingface.co/$1/$2
