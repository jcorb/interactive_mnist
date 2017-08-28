# Interactive MNIST viewer

This is an attempt to create an "operational" machine learning application.  The classifier was pre-trained using TensorFlow on the MNIST digits dataset.  This app allows you to draw a digit and submit it to the classifier, which then performs the classifier and displays what it thinks it is.  

It is written in Python and uses the Bokeh library to handle the interactive drawing and display.  The Bokeh library wasn't really designed with this sort of drawing in mind, so it is a bit of a hack, but seems to work OK, if a little clunky. 

## Requirements

* Python 2.7
* Bokeh 
* Pandas  
* numpy
* scipy
* tensorflow


## Usage
### To run locally:

Download the repo.


To run the tool enter: `bokeh serve --show ` into the command line from the parent directory.

This will launch a web browser and display the app.  To use, click on the top plot in order to make a digit (i.e numbers 0-9).  To select multiple points hold "shift" while you click.  Make the digit take up at least three-quarters or so of the height. Clicking "reset" will reset the top plot.  One you have a digit drawn, click "classify".  After a few seconds, the image you drew and the digit that the classifier thinks it is will be displayed.

### To access via the internet:

The app is running on Heroku and can be accessed here https://calm-meadow-39640.herokuapp.com/app
 
