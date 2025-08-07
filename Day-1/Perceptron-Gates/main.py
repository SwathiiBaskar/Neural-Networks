from Perceptron import Perceptron
from gates import get_gates_data
import matplotlib.pyplot as plt
import numpy as np

def run_gate(gate_name):
    x,y=get_gates_data(gate_name)
    if(y is None):
        print("Unsupported gate")
        return
    #creating object for Perceptron class
    p=Perceptron(input_size=2)
    #training officially
    p.train(x,y,epochs=10)
    print(f"Testing {gate_name.upper()} Gate: ")
    for xi in x:
        #xi(s) are feature vectors
        print(f"Input {xi}, Predicted: {p.predict(xi)}")
    plot_boundary(p,x,y)

def plot_boundary(model,x,y):
    for xi,target in zip(x,y):
        color='red' if target==0 else 'blue'
        plt.scatter(xi[0],xi[1],c=color)
    x1=np.linspace(-0.1,1.1,100)
    x2=-(model.weights[0] * x1 +model.bias)/model.weights[1]  #Decision Boundary -> a line/curve that separates classes
    #the perceptron decides between/among class and create the decision boundary
    """w₁*x₁+w₂*x₂+b = 0
    → x₂ = -(w₁*x₁ + b)/w₂
    """
    plt.plot(x1,x2,label="Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title(f"Perceptron decision boundary")
    plt.grid(True)
    plt.show()
    
def main():
    while True:
        print("Choose a logic gate to stimulate: ")
        print("1. AND")
        print("2. OR")
        print("3. NAND")
        print("4. NOR")
        print("0. Exit")

        choice=input("Enter your choice: ")
        gate_map = {
            "1": "AND",
            "2": "OR",
            "3": "NAND",
            "4": "NOR"
        }
        if(choice=="0"):
            break
        elif choice in gate_map:
            run_gate(gate_map[choice])
        else:
            print("Invalid choice. Try again!")

if __name__=="__main__":
    main()
