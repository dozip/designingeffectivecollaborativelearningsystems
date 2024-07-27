# Setup
In order to start the back- and frontend properly, some things have to be done first

## Installation requirements
### pipenv 
virtual enviroment for the python backend (flask)

run `$ pip install pipenv`

### Node.js 
JavaScript runtime enviroment

https://nodejs.org/de

## Installing the node modules
Navigate to `frontend\gui`

run `$ npm install`

# Starting the GUI
## Starting the backend
Open a new terminal and navigate to `backend\api`

run `$ pipenv shell`

run `$ python api.py`

The backend should then be running at **http://localhost:5000**

## Starting the frontend
Open a new terminal and navigate to `frontend\gui`

run `$ npm run serve` - notice that this is dev-mode only

the frontend should then be running at **http://localhost:8080**
