# Install dependencies
apt-get install python3 python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools -y
apt-get install python3-venv -y
apt-get install nginx -y

# Start and enable Nginx
systemctl start nginx
systemctl enable nginx

# Set up virtual environment and run Flask app

python3 -m venv venv
source venv/bin/activate

pip install requirements.txt
pip install gunicorn
pip install wheel
pip install gunicorn flask
# Press CTRL+C to close the application.

# Create wsgi.py using echo
echo 'from app6 import app
if __name__ == "__main__":
    app.run()' > wsgi.py
# Create wsgi.py using echo
# Start Gunicorn

venv/bin/gunicorn --bind 0.0.0.0:5000 wsgi:app


# Deactivate virtual environment
deactivate

# Create systemd service file for Flask using echo
echo '[Unit]
Description=Gunicorn instance to serve Flask
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=/root/project
Environment="PATH=/root/project/venv/bin"
ExecStart=/root/project/venv/bin/gunicorn --bind 0.0.0.0:5000 wsgi:app

[Install]
WantedBy=multi-user.target' | sudo tee /etc/systemd/system/flask.service


# Set permissions and reload systemd
cd ..
sudo chown -R root:www-data hypertension
sudo chmod -R 775 hypertension
systemctl daemon-reload

# Start and enable Flask service
systemctl start flask
systemctl enable flask
systemctl status flask

# Configure Nginx using echo
echo 'server {
    listen 80;
    server_name flask.example.com;
    location / {
        include proxy_params;
        proxy_pass  http://127.0.0.1:5000;
    }
}' > /etc/nginx/conf.d/flask.conf

# Test Nginx configuration
nginx -t

# Restart Nginx
systemctl restart nginx
