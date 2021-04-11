import matplotlib.pyplot as plt
import numpy as np
import tensorslow as ts

# Create two clusters of red points centered at (0, 0) and (1, 1), respectively.
red_points = np.concatenate((
    0.2*np.random.randn(25, 2) + np.array([[0, 0]]*25),
    0.2*np.random.randn(25, 2) + np.array([[1, 1]]*25)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blue_points = np.concatenate((
    0.2*np.random.randn(25, 2) + np.array([[0, 1]]*25),
    0.2*np.random.randn(25, 2) + np.array([[1, 0]]*25)
))

# Plot them
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show()

# Create a new graph
graph = ts.Graph().as_default()

# Create training input placeholder
X = ts.placeholder()

# Create placeholder for the training classes
c = ts.placeholder()

# Build a hidden layer
W_hidden = ts.Variable(np.random.randn(2, 2))
b_hidden = ts.Variable(np.random.randn(2))
p_hidden = ts.sigmoid(ts.add(ts.matmul(X, W_hidden), b_hidden))

# Build the output layer
W_output = ts.Variable(np.random.randn(2, 2))
b_output = ts.Variable(np.random.randn(2))
p_output = ts.softmax(ts.add(ts.matmul(p_hidden, W_output), b_output))

# Build cross-entropy loss
J = ts.negative(ts.reduce_sum(ts.reduce_sum(ts.multiply(c, ts.log(p_output)), axis=1)))

# Build minimization op
minimization_op = ts.train.GradientDescentOptimizer(learning_rate=0.03).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

# Create session
session = ts.Session()

# Perform 100 gradient descent steps
for step in range(1000):
    J_value = session.run(J, feed_dict)
    if step % 100 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)

graph.export_dot('mlp_')

# Print final result
W_hidden_value = session.run(W_hidden)
print("Hidden layer weight matrix:\n", W_hidden_value)
b_hidden_value = session.run(b_hidden)
print("Hidden layer bias:\n", b_hidden_value)
W_output_value = session.run(W_output)
print("Output layer weight matrix:\n", W_output_value)
b_output_value = session.run(b_output)
print("Output layer bias:\n", b_output_value)

# Visualize classification boundary
xs = np.linspace(-2, 2)
ys = np.linspace(-2, 2)
pred_classes = []
for x in xs:
    for y in ys:
        pred_class = session.run(p_output,
                              feed_dict={X: [[x, y]]})[0]
        pred_classes.append((x, y, pred_class.argmax()))
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
plt.show()