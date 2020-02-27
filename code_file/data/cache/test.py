import pickle
data = pickle.load(open('Train_action_target.pkl','rb'))
print(type(data))
print(len(data))
print(data[0:10])