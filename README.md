# TensorSlow
## A re-implementation of <a href="http://www.tensorflow.org">TensorFlow</a> functionality in pure python

TensorSlow is a minimalist machine learning API that mimicks the TensorFlow API, but is implemented in pure python (without a C backend). The source code has been built with maximal understandability in mind, rather than maximal efficiency. Therefore, TensorSlow should be used solely for educational purposes. If you want to understand how deep learning libraries like TensorFlow work under the hood, this may be your best shot. 

I have written an article in my blog at <a href="http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/">deepideas.net</a> that develops this library step by step, explaining all the math and algorithms on the way: <a href="http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/">Deep Learning From Scratch</a>.

## How to use
Import:

    import tensorslow as ts

Create a computational graph:

    ts.Graph().as_default()

Create input placeholders:

    training_features = ts.placeholder()
    training_classes = ts.placeholder()

Build a model:

	weights = ts.Variable(np.random.randn(2, 2))
	biases = ts.Variable(np.random.randn(2))
	model = ts.softmax(ts.add(ts.matmul(X, W), b))

Create training criterion:

    loss = ts.negative(ts.reduce_sum(ts.reduce_sum(ts.multiply(training_classes, ts.log(model)), axis=1)))

Create optimizer:

    optimizer = ts.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

Create placeholder inputs:

	feed_dict = {
		training_features: my_training_features,
		training_classes: my_training_classes
	}

Create session:

	session = ts.Session()

Train:

	for step in range(100):
		loss_value = session.run(loss, feed_dict)
		if step % 10 == 0:
			print("Step:", step, " Loss:", loss_value)
		session.run(optimizer, feed_dict)

Retrieve model parameters:

	weights_value = session.run(weigths)
	biases_value = session.run(biases)

Check out the `examples` directory for more.
