# Intelligent Survivor Locator

In the case of a natural calamity, the main problem which is faced by victims, is the lack of electricity, lack of battery life on phones and limited mobile connectivity ( because of damaged mobile towers ). Due to this, they find it difficult to communicate their whereabouts to officials who can help them. The lack of support from officials leaves them in a dire need of necessities such as food, water and medicines.  

## Proposed Solution: 
Our solution aims at first surveying the area affected by the natural disaster, in order to quantify the number of people stuck in the area and provide them with essential supplies, in the most efficient way, at the earliest. Our system focuses on distinguishing areas based on the level of damage using images of the damaged areas from drones. Various areas can be put in different categories based on the extent of damage to the people. This data can be then used to prioritise their assistance to different areas by connecting to the nearest relief camps.

## Design Flow 



### Instructions to run the code locally

1. Clone the repository.
2. Create a python virtual environment from the root directory and activate it.
```
$ virtualenv -p python3 venv
$ . venv/bin/activate 
```
3. Install the requirements specified in requirements.txt 
```
$ pip install -r requirements.txt 
```

4. Run the demo. 
```
python3 demo.py
``` 

### Instructions to run the web app.

1. Change directory to web. 
``` 
$ cd web
$ python main.py
 ```
2. Create a python virtual environment from the root directory and activate it.
```
$ virtualenv -p python3 venv
$ . venv/bin/activate 
```
3. Install the requirements specified in requirements.txt 
```
$ pip install -r requirements.txt 
```

4. Run the webapp. 
```
python3 main.py
``` 



