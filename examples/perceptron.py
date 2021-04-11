import matplotlib.pyplot as plt
import numpy as np
import tensorslow as ts

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2*np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2*np.ones((50, 2))

# Create a new graph
graph=ts.Graph().as_default()

X = ts.placeholder()
c = ts.placeholder()

# Initialize weights randomly
W = ts.Variable(np.random.randn(2, 2))
b = ts.Variable(np.random.randn(2))

# Build perceptron
p = ts.softmax(ts.add(ts.matmul(X, W), b))

# Build cross-entropy loss
J = ts.negative(ts.reduce_sum(ts.reduce_sum(ts.multiply(c, ts.log(p)), axis=1)))

# Build minimization op
minimization_op = ts.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

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
for step in range(100):
    J_value = session.run(J, feed_dict)
    if step % 10 == 0:
        print("Step:", step, " Loss:", J_value)
    session.run(minimization_op, feed_dict)

# Print final result
W_value = session.run(W)
print("Weight matrix:\n", W_value)
b_value = session.run(b)
print("Bias:\n", b_value)

graph.export_dot()

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -W_value[0][0]/W_value[1][0] * x_axis - b_value[0]/W_value[1][0]
plt.plot(x_axis, y_axis)

# Add the red and blue points
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')
plt.show()