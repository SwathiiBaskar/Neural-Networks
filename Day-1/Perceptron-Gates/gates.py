import numpy as np

def get_gates_data(gate_name):
    #input array
    x=np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])
    #target labels
    gate_output={
        "AND": np.array([0,0,0,1]),
        "OR": np.array([0,1,1,1]),
        "NAND": np.array([1,1,1,0]),
        "NOR": np.array([1,0,0,0])
        }
    return x, gate_output.get(gate_name.upper()) #returns a tuple (input array, target labels)
                                                                          #get recovers the value of the key specified
