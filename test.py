import json
import numpy as np

# X_train = np.array([[1,2,3,4],[5,6,7,8]])
# print(X_train.shape)
# X_train = X_train[:, np.newaxis]
# print(X_train)
# print(X_train.shape)
train_config = None
test_config = 'df'
train_config = train_config or test_config
print(train_config)

data = {
    'name' : 'ACME',
    'shares' : 100,
    'price' : 542.23
}
if 'name' in data:
	print('hello')
print(data.keys())
print(data.get('happy', True))


train_config = {}
keep_model_in_mem = train_config.get("keep_model_in_mem", True)
random_state = train_config.get("random_state", 0)
phases = train_config.get("phases", ["train", "test"])
print(keep_model_in_mem, random_state, phases)

look_indexs = {'a':1, 'b':2}
for _i, i in look_indexs.items():
	print(_i,i)

y_probas = []
y_proba_cv = np.zeros((60000,10))
y_probas.append(y_proba_cv)
val_idx = np.array([0, 1, 2])
print(y_probas[0][val_idx, :])


y_train_proba_li = np.zeros((2, 2))
y_proba = np.array([[1,5],[2,1
	]])
x = np.array([1,2])
y_train_proba_li += y_proba
y_train_proba_li /= 5
print(y_train_proba_li)
print(np.argmax(y_train_proba_li, axis=1))
print(np.argmax(x, axis=0))

my_matrix = np.loadtxt(open("/home/jluo80/Desktop/SF/Python method/deep_learning/train.csv","rb"),dtype=np.str,delimiter=",",skiprows=1)
print(my_matrix)

my_matrix2 = np.loadtxt(open("/home/jluo80/Desktop/training_1.csv","rb"),delimiter=",",skiprows=1)
print(my_matrix2)
print(my_matrix2.shape)

X_train = my_matrix2[:, :-1]
y_train = my_matrix2[:, -1]

print('X_train', X_train)
print('y_train', y_train)

print(np.unique(y_train))


if True:
	pass
	print('dfd')
elif True:
	print('df')