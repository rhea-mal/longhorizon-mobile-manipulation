import pickle
import pdb
with open("demo00000.pkl", "rb") as f:
    demo = pickle.load(f)
    for step in demo:
        # pdb.set_trace()
        print(step["click"])