sudo apt-get install git-lfs
sudo rm -R ../../chat-ui
cd ../.. && git clone https://huggingface.co/spaces/huggingchat/chat-ui
cd ./chat-ui && npm install
cp -v ../AI/data/env.local .env.local