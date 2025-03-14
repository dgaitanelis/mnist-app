# MNIST Digit Recognizer

This is a web application built with Streamlit, PyTorch, PostgreSQL, and Docker. It recognizes handwritten digits from the MNIST dataset.

## Live Application
The live application is hosted on Google Cloud. You can access it here:
([http://34.105.147.212:8501/](http://34.142.24.224:8501/))

## Technologies Used
1. Streamlit: Frontend for the web application.
2. PyTorch: Model for handwritten digit recognition.
3. PostgreSQL: Database to store predictions and feedback.
4. Docker: Containerization for easy setup and deployment.

## Important Note

To run the app, make sure Docker and Docker Compose are installed.

## Running the App Locally

1. Clone the repository:
	git clone https://github.com/dgaitanelis/mnist-app.git
	cd mnist-app

2. Install the required dependencies:
	pip install -r requirements.txt

3. Build and run the app using Docker: Make sure Docker and Docker Compose are installed. Then, run:
	sudo docker-compose up --build -d

4. Access the app: Once the containers are running, open your browser and go to:
	http://localhost:8501 (for local use on your machine)
	[http://34.105.147.212:8501/](http://34.142.24.224:8501/) (for remote access - if hosted on a server like Google Cloud)

5. Stop the application: To stop the app, use:
   sudo docker-compose down
