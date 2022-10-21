import json
import matplotlib.pyplot as plt

with open('misc/output_first2epochs.out') as file:
    lines = file.readlines()

lines = [line.rstrip() for line in lines]

eval_loss = []
eval_epochs = []
eval_kappa = []
train_loss = []
train_epochs = []

for line in lines:
    if "{'eval_loss'" in line:
        index = line.index("{'eval_loss'")
        dic_str = line[index:].replace("'", '"')
        dict = json.loads(dic_str)
        eval_loss.append(dict['eval_loss'])
        eval_kappa.append(dict['eval_kappa'])
        eval_epochs.append(dict['epoch'])

    elif "{'loss'" in line:
        index = line.index("{'loss'")
        dic_str = line[index:].replace("'", '"')
        dict = json.loads(dic_str)
        train_loss.append(dict['loss'])
        train_epochs.append(dict['epoch'])


with open('misc/output_full_run.out') as file:
    lines = file.readlines()

lines = [line.rstrip() for line in lines]

for line in lines:
    if "{'eval_loss'" in line:
        index = line.index("{'eval_loss'")
        dic_str = line[index:].replace("'", '"')
        dict = json.loads(dic_str)

        if dict['epoch'] > 2:
            eval_loss.append(dict['eval_loss'])
            eval_kappa.append(dict['eval_kappa'])
            eval_epochs.append(dict['epoch'])

    elif "{'loss'" in line:
        index = line.index("{'loss'")
        dic_str = line[index:].replace("'", '"')
        dict = json.loads(dic_str)
        if dict['epoch'] > 2:
            train_loss.append(dict['loss'])
            train_epochs.append(dict['epoch'])


plt.plot(eval_epochs, eval_loss, color='m', label='val loss')
plt.plot(train_epochs, train_loss, linestyle='solid', color='tab:green', label='train loss')
plt.plot(eval_epochs, eval_kappa, linestyle='dotted', label='$\kappa$')
plt.xlabel('epoch')
plt.title('Training progression')
plt.legend()
plt.show()