sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.8 python3.8-venv -y
python3.8 -m venv myenv38
source myenv38/bin/activate
pip install -r requirements.txt
# Install dependencies
python app6.py
