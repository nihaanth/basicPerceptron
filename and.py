from utils.model import Perceptron
from all_utils import prepare_data

AND = {
    'x1':[0,0,1,1],
    'x2':[0,1,0,1],
    'y':[0,0,0,1],  
}

df = pd.DataFrame(AND)

print(df)

model = Perceptron(eta=ETA,epochs=EPHOCH)
model.fit(X,y)

if __name__ == '__main__':#entry point
    main()