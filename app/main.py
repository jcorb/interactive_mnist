import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Div
from bokeh.layouts import column, widgetbox, row
from bokeh.models.widgets import Button
from bokeh.io import export_png
from scipy.misc import imread, imresize
from scipy import ndimage
import pickle
import os
import tensorflow as tf

def load_parameters():
    '''Load the trained parameters'''
    pickle_file = os.path.join(os.path.dirname(__file__), 'data', 'tf_parameters.pkl')
    with open(pickle_file, 'rb') as f:
        parameters = pickle.load(f)

    return parameters

def compute_relu(W, A, b):
    """Compute LINEAR-> RELU activation"""
    Z = tf.add(tf.matmul(W, A), b)
    A = tf.nn.relu(Z)

    return A

def forward_propagation(X, parameters, n_L):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> ... -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", ... , "W n_L", "b n_L"
                  the shapes are given in initialize_parameters

    Returns:
    ZL -- the output of the last LINEAR unit
    """

    for layer in range(n_L):

        if layer == 0:
            A_prev = X
        else:
            A_prev = A

        if layer != n_L - 1:
            A = compute_relu(parameters['W' + str(layer + 1)], A_prev, parameters['b' + str(layer + 1)])
        else:
            ZL = tf.add(tf.matmul(parameters['W' + str(layer + 1)], A_prev), parameters['b' + str(layer + 1)])

    return ZL

def predict(X, parameters, n_L):
    """Perform the forward propagation on the input image using the trained parameters"""
    params = {}
    for key in parameters.keys():
        params[key] = tf.convert_to_tensor(parameters[key])

    x = tf.placeholder("float", [X.size, 1])

    zL = forward_propagation(x, params, n_L)
    p = tf.argmax(zL)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})
    sess.close()
    return prediction

def process_image(image):
    '''Takes the image of the digit and prepares it for use in the classifier'''

    #center the image
    i_c, j_c = ndimage.measurements.center_of_mass(image)
    i_shift = np.floor(120 - i_c)
    j_shift = np.floor(120 - j_c)
    image_centered = ndimage.shift(image, [-i_shift, -j_shift], mode='constant', cval=255)

    # resize the image to be 28 x 28 pixels
    image = imresize(image_centered, (28, 28))
    # put the image into a vector and reverse the colors
    image = np.abs(255. - image).ravel()
    image = image.reshape((image.size, 1))

    return image

def get_image():

    #save the graph as a png file
    image_file = os.path.join(os.path.dirname(__file__), 'images_out', 'image.png')
    export_png(layout, image_file)

    #load the png file and trim the unnecessary edges
    img_in = imread(image_file)
    img = img_in[5:245, 30:270, 0]

    img = process_image(img)
    classified_digit = predict(img/255., parameters, n_L=3)
    class_cds.data = dict(x=[0], y=[0], digit=[str(classified_digit[0])],
                          image=[np.flipud(img.reshape(28,28))])
    return

global parameters
parameters = load_parameters()


p = figure(plot_width=300, plot_height=250, tools='tap, reset')
p.axis.visible = False
p.grid.visible = False

X, Y = np.meshgrid(np.arange(0,15), np.arange(0,15))
r = p.circle(X.ravel(), Y.ravel(),
             color='black',
                nonselection_color='white',
             size=18)

class_cds = ColumnDataSource(data=dict(x=[0], y=[0], digit=['_'],
                                       image=[np.zeros((28,28), dtype=np.uint8)]))
pclass = figure(plot_width=150, plot_height=150, tools='',
                x_range = (-10, 20), y_range = (-10, 20), toolbar_location=None)
pclass.axis.visible = False
pclass.grid.visible = False
pclass.text('x', 'y', 'digit', source=class_cds, text_color='tomato', text_font_size="50pt")

pimage = figure(plot_width=150, plot_height=150, tools='',
                x_range = (0, 28), y_range = (0, 28), toolbar_location=None)
pimage.axis.visible = False
pimage.grid.visible = False
pimage.image('image', 'x', 'y', dw=28, dh=28, source=class_cds, palette="Greys256")

classify = Button(label='Classify')
classify.on_click(get_image)

div = Div(text="""<h1>Interactive MNIST classifier</h1>   
                    <h3>Instructions</h3> 
                    Use the tap tool to select points and draw digits on the above canvas (hold "shift" to select mulitple points).  
                    When you are done, \
                    click "classify".  An image of the digit you have drawn, and the result of the classifier 
                    will appear in the lower two plots (it takes a few seconds).  Hopefully it gets it right!  The classifier is a 3-layer TensorFlow neural
                    net trained using the MNIST digit dataset.
                    """,
          width=300, height=600)

layout = column(p, widgetbox(classify), row(pimage, pclass), widgetbox(div))
curdoc().add_root(layout)
curdoc().title = "Interactive MNIST"
