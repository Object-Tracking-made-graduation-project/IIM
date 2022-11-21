echo "Creating a new virtual environment venv"
call python -m venv venv
call venv/Scripts/activate.bat
echo "The virtual environment venv has been activated"
call pip install -r requirements.txt
echo "All required packages were successfully installed. Please press any key to continue..."
pause