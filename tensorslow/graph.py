class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self
        return _default_graph

    def export_dot(self,filename_prefix_='graph_'):
        def custom_hash(obj):
            outputstr = str(hash(obj)).replace('-', '_')
            if isinstance(obj, Operation):
                cls = str(obj.__class__)
                return cls.split('.')[-1].split('\'')[0] + '_' + outputstr
            return outputstr
        format = ["digraph {"]
        nvar, nop = len(self.variables),len(self.operations)
        for idvar,var in enumerate(self.variables):
            varstr = 'var_'+str(idvar) #str(custom_hash(var))
            format.append('{0} [label="var/{1}",color=blue,shape=box]'.format(varstr,idvar))
            for idx,output in enumerate(var.consumers):
                if isinstance(output, Operation):
                    outputstr = 'op_' + str(custom_hash(output))
                else:
                    outputstr = 'var_' + str(custom_hash(output))
                format.append('{0} -> {1} [label="{2} {3}"]'.format(varstr,outputstr,idx,var))

        for idop,op in enumerate(self.operations):
            opstr = 'op_'+str(custom_hash(op))
            format.append('{0} [label="{0}/{1}"]'.format(opstr,idop))
            for idx,output in enumerate(op.consumers):
                if isinstance(output,Operation):
                    outputstr = 'op_'+str(custom_hash(output))
                else:
                    outputstr = 'var_' + str(custom_hash(output))
                format.append('{0} -> {1} [label="{4}/{2} {3}"]'.format(opstr,outputstr,idx,opstr,idop))

        format.append("}")
        dot_str = "\n".join(format)
        with open(filename_prefix_+'{0}var_{1}op.dot'.format(nvar,nop),'w') as fp:
            fp.write(dot_str)
        return dot_str

class Operation:
    """Represents a graph node that performs a computation.

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=[]):
        """Construct Operation
        """
        self.input_nodes = input_nodes

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def compute(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass


class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
        self.consumers = []

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)


class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.consumers = []

        # Append this variable to the list of variables in the currently active default graph
        _default_graph.variables.append(self)
