#!/bin/bash
echo "Starting deployment script..."

cd /home/ubuntu/AI-Assistant/inrellidocsproject

# Activate virtual environment
source /home/ubuntu/AI-Assistant/myenv/bin/activate

# Run migrations
echo "Applying Django migrations..."
python3 manage.py migrate --noinput

# Restart Gunicorn
echo "Restarting Gunicorn service..."
sudo systemctl restart gunicorn

echo "Deployment completed successfully!"
