@ECHO OFF
ECHO =======================================
ECHO  Starting the Dashboard Application...
ECHO  This may take a minute the first time.
ECHO =======================================

docker-compose up

ECHO.
ECHO ==========================================================
ECHO  Application started successfully!
ECHO.
ECHO  Please open your web browser and go to:
ECHO  http://localhost:8050
ECHO ==========================================================
ECHO.
PAUSE