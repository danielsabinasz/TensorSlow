from tensorslow.graph import Operation
from tensorslow.graph import Variable
from tensorslow.gradients import _gradient_registry

class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):
                # Compute gradients
                grad_table = compute_gradients(loss)

                # Iterate all variables
                for node in grad_table:
                    if type(node) == Variable:
                        # Retrieve gradient for this variable
                        grad = grad_table[node]

                        # Take a step along the direction of the negative gradient
                        node.value -= learning_rate * grad

        return MinimizationOperation()


from queue import Queue


def compute_gradients(loss):
    # grad_table[node] will contain the gradient of the loss w.r.t. the node's output
    grad_table = {}

    # The gradient of the loss with respect to the loss is just 1
    grad_table[loss] = 1

    # Perform a breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # If this node is not the loss
        if node != loss:
            #
            # Compute the gradient of the loss with respect to this node's output
            #
            grad_table[node] = 0

            # Iterate all consumers
            for consumer in node.consumers:

                # Retrieve the gradient of the loss w.r.t. consumer's output
                lossgrad_wrt_consumer_output = grad_table[consumer]

                # Retrieve the function which computes gradients with respect to
                # consumer's inputs given gradients with respect to consumer's output.
                consumer_op_type = consumer.__class__
                bprop = _gradient_registry[consumer_op_type]

                # Get the gradient of the loss with respect to all of consumer's inputs
                lossgrads_wrt_consumer_inputs = bprop(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    # If there is a single input node to the consumer, lossgrads_wrt_consumer_inputs is a scalar
                    grad_table[node] += lossgrads_wrt_consumer_inputs

                else:
                    # Otherwise, lossgrads_wrt_consumer_inputs is an array of gradients for each input node

                    # Retrieve the index of node in consumer's inputs
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)

                    # Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    # Add to total gradient
                    grad_table[node] += lossgrad_wrt_node

        #
        # Append each input node to the queue
        #
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    # Return gradients for each visited node
    return grad_table
