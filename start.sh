#!/bin/bash
echo "======================================="
echo " Starting the Dashboard Application..."
echo " This may take a minute the first time."
echo "======================================="

docker-compose up

echo ""
echo "=========================================================="
echo " ✅ Application started successfully!"
echo ""
echo " Please open your web browser and go to:"
echo " http://localhost:8050"
echo "=========================================================="