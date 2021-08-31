'''
This script exctracts training variables from all logs from 
tensorflow event files ("event*"), writes them to Pandas 
and finally stores in long-format to a CSV-file including
all (readable) runs of the logging directory.
The magic "5" infers there are only the following v.tags:
[lr, loss, acc, val_loss, val_acc]
'''

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Get all event* runs from logging_dir subdirectories
# PATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/FULL-VIDEOS/mobilenet_lstm_experiment_decoder_5"

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y = np.array(y)
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def tabulate_events(dpath):
    events = ['train', 'validation']
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in events]

    tags = summary_iterators[0].Tags()['scalars']

    # for it in summary_iterators:
    #     assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        if not tag.__contains__('batch'):
            steps = [e.step for e in summary_iterators[1].Scalars(tag)]

            for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
                assert len(set(e.step for e in events)) == 1

                out[tag].append([e.value for e in events])

    return out, steps


def logs2dataframe(logging_dir):
    data_loss = pd.DataFrame()
    data_jacc = pd.DataFrame()
    data_task_loss = pd.DataFrame()

    dirs = ['train', 'validation']
    d, steps = tabulate_events(logging_dir)
    tags, values = zip(*d.items())
    np_values = np.array(values)
    column_names = dirs + ['step']

    for index, tag in enumerate(tags):
        epochs = np.arange(0, len(np_values[index]), 1)
        data = np.concatenate((np_values[index], epochs.reshape(len(epochs), 1)), axis=1)
        df = pd.DataFrame(data, index=steps, columns=column_names)
        df['metric'] = tag
        if tag.__contains__('epoch_loss'):
            data_loss = pd.concat([data_loss, df])
        elif tag.__contains__('loss'):
            data_task_loss = pd.concat([data_task_loss, df])
        elif tag.__contains__('sparse'):
            data_jacc = pd.concat([data_jacc, df])
    return data_loss, data_task_loss, data_jacc


def plot_logs(data_loss, data_jacc, name='plot'):
    val_color = (0 / 255, 128 / 255, 255 / 255)
    train_color = (255 / 255, 128 / 255, 0 / 255)
    ### Plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(data_loss.step, data_loss.train,
             color=train_color, alpha=0.3)
    plt.plot(data_loss.step, smooth(data_loss.train, 50),
             color=train_color, alpha=0.8)

    plt.plot(data_loss.step, data_loss.validation,
             color=val_color, alpha=0.3)
    plt.plot(data_loss.step, smooth(data_loss.validation, 50),
             color=val_color, alpha=0.8)

    plt.ylim([0, 0.2])
    plt.legend(["Train", "", "Validation", ""])
    plt.xlabel("Epochs")
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(data_jacc.step, data_jacc.train, color=train_color,
             alpha=0.3)
    plt.plot(data_jacc.step, smooth(data_jacc.train, 50),
             color=train_color, alpha=0.8)

    plt.plot(data_jacc.step, data_jacc.validation,
             color=val_color, alpha=0.3)
    plt.plot(data_jacc.step, smooth(data_jacc.validation, 50),
             color=val_color, alpha=0.8)

    plt.title("Jaccard Index")
    plt.legend(["Train", "", "Validation", ""])
    plt.xlabel("Epochs")
    plt.show()

    ## PLOT 2
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(data_loss.step, data_loss.train,
            color=train_color, alpha=0.1)

    ax.plot(data_loss.step, smooth(data_loss.train, 50),
            color=train_color, alpha=0.8)

    ax.plot(data_loss.step, data_loss.validation,
            color=val_color, alpha=0.1)
    ax.plot(data_loss.step, smooth(data_loss.validation, 50),
            color=val_color, alpha=0.8)
    # set x-axis label
    ax.set_xlabel("Epochs", fontsize=14)
    # set y-axis label
    ax.set_ylabel("Loss", fontsize=14)
    ax.set(ylim=[0, 0.5])
    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object

    ax2.plot(data_jacc.step, data_jacc.train, color=train_color,
             alpha=0.1)
    ax2.plot(data_jacc.step, smooth(data_jacc.train, 50),
             color=train_color, alpha=0.8)
    ax2.plot(data_jacc.step, data_jacc.validation,
             color=val_color, alpha=0.1)
    ax2.plot(data_jacc.step, smooth(data_jacc.validation, 50),
             color=val_color, alpha=0.8)
    ax2.set(ylim=[0, 1])
    ax2.set_ylabel("Jaccard Index", fontsize=14)

    plt.legend(["Train", "", "Validation", ""], loc='upper right',
               bbox_to_anchor=(1.35, 1.05))
    plt.title(name)
    plt.show()


# Get all event* runs from logging_dir subdirectories
# PATH = "C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/IMAGE-PROCESSING/FULL-VIDEOS"
PATH = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/"
# experiments = glob.glob(os.path.join(PATH, 'mobilenet_lstm_experiment_*'))
experiments = glob.glob(os.path.join(PATH, 'mobilenet_lstm_masks_experiment_*'))
experiments.sort()
legend = []
output_path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/plots"
epochs = 2000
# tips_1 = 1725
# tips_2 = 1000
# tips_3 = 1650
# tips_4 = 1163
# tips_5 = 1625
#
# masks_1 = 2000
# masks_2 = 1700
# masks_3 = 1875
# masks_4 = 1225
# masks_4 = 1975
tips_1 = [1725, 1000, 1650, 1163, 1625]
masks_1 = [2000, 1700, 1875, 1225, 1975]
tips_2 = [2000, 1563, 350, 1338, 800]
masks_2 = [1950, 1900, 1675, 1475, 200]
colors = ["#2414FF", "#FF2424", "#4FA13D", "#DA2BED", "#FF830F"]
i = 0
fig = plt.figure(figsize=(15, 8))
for exp in experiments:
    if not exp.__contains__('colab'):
        exp_num = np.int(exp.split('_')[-1])
        if exp_num>1 or exp.__contains__('decoder'):
            if exp.__contains__('decoder'):
                # b = tips_1[exp_num - 1]
                b = masks_1[exp_num - 1]
                mode = 'decoder'
            else:
                # b = tips_2[exp_num - 1]
                b = masks_2[exp_num - 1]
                mode = 'fine-tune'
            logging_dir = os.path.join(exp, 'logs')
            data_loss, data_task_loss, data_jacc = logs2dataframe(logging_dir)
            if b < epochs:
                aux1 = data_loss.iloc[:b]
                aux2 = data_loss.iloc[-(epochs - b):]
                aux2['step'] = aux2['step'] + (epochs - np.max(aux2['step']))
                aux = [aux1, aux2]
                data_loss = pd.concat(aux)

                # aux1 = data_task_loss.iloc[:b]
                # aux2 = data_task_loss.iloc[-(epochs - b):]
                # aux2['step'] = aux2['step'] + (epochs - max(aux2['step']))
                # aux = [aux1, aux2]
                # data_task_loss = pd.concat(aux)

                aux1 = data_jacc.iloc[:b]
                aux2 = data_jacc.iloc[-(epochs - b):]
                aux2['step'] = aux2['step'] + (epochs - np.max(aux2['step']))
                aux = [aux1, aux2]
                data_jacc = pd.concat(aux)
            print(len(data_loss))
            print(np.max(data_loss['step']))
            data_loss['Mode'] = mode
            data_loss['Total_epochs'] = data_loss['step']
            data_loss.to_csv(os.path.join(exp, "epoch_loss_decoder.csv"))
            data_jacc['Mode'] = mode
            data_jacc['Total_epochs'] = data_jacc['step']
            data_jacc.to_csv(os.path.join(exp, "1_jaccard_sparse3D_decoder.csv"))

    plt.subplot(3, 1, 1)
    # plt.plot(np.array(data_loss.validation), alpha=0.4)
    val = smooth(data_loss.validation, 100)
    plt.plot(val[:1950], alpha=1, color=colors[i])
    plt.subplot(3, 1, 2)
    # plt.plot(np.array(data_loss.validation), alpha=0.4)
    val = smooth(data_loss.train, 100)
    plt.plot(val[:1950], alpha=1, color=colors[i])
    plt.subplot(3, 1, 3)
    # plt.plot(np.array(data_jacc.validation), alpha=0.4)
    val = smooth(data_jacc.validation, 100)
    plt.plot(val[:1950], alpha=1, color=colors[i])
    val = smooth(data_jacc.train, 100)
    plt.plot(val[:1950], alpha=1, color=colors[i], linestyle='dashed')
    # name = 'BT' + exp.split('mobilenet_lstm_experiment_decoder_')[-1]
    name = 'B' + exp.split('mobilenet_lstm_masks_experiment_decoder_')[-1]
    legend.append(name)
    i += 1
    # legend.append("")
    # plot_logs(data_loss, data_jacc, name=name)
plt.subplot(3, 1, 1)
plt.xlabel('Epochs', fontsize=12)
plt.ylim([0, 0.2])
plt.xlim([0, 2000])
plt.ylabel('Loss function (validation)', fontsize=12)
plt.legend(legend, loc='upper right', bbox_to_anchor=(1.12, 1.0), fontsize=12)

plt.subplot(3, 1, 2)
plt.xlabel('Epochs', fontsize=12)
# plt.ylim([0, 0.02])
plt.xlim([0, 2000])
plt.ylabel('Loss function (train)', fontsize=12)

plt.subplot(3, 1, 3)
plt.ylabel('Jaccard index', fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.xlim([0, 2000])
plt.ylim([0, 0.8])
plt.legend(["Validation", "Train"])
fig.savefig(os.path.join(output_path, 'b.pdf'), format='pdf', dpi=500, transparence=True)
fig.savefig(os.path.join(output_path, 'b.png'), format='png', dpi=500)
# fig.savefig(os.path.join(output_path, 'bt.pdf'), format='pdf', dpi=500, transparence=True)
# fig.savefig(os.path.join(output_path, 'bt.png'), format='png', dpi=500)
plt.show()

##
PATH = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/"
# folder_name = 'mobilenet_lstm_experiment'
folder_name = 'mobilenet_lstm_masks_experiment'
for i in range(1, 6):
    decoder = pd.read_csv(os.path.join(PATH, folder_name + '_decoder_{}'.format(i), 'epoch_loss_decoder.csv'))
    decoder = decoder[(['train', 'validation', 'Mode', 'Total_epochs'])]
    finetune = pd.read_csv(os.path.join(PATH, folder_name + '_{}'.format(i), 'epoch_loss_decoder.csv'))
    finetune = finetune[(['train', 'validation', 'Mode', 'Total_epochs'])]
    last_epoch_decoder = np.max(decoder['Total_epochs'])
    finetune['Total_epochs'] = finetune['Total_epochs'] + last_epoch_decoder
    aux = pd.concat([decoder, finetune])
    aux.to_csv(os.path.join(PATH, 'experiments_logs', 'epoch_loss_' + folder_name + '_{}.csv'.format(i)))

    decoder = pd.read_csv(os.path.join(PATH, folder_name + '_decoder_{}'.format(i), '1_jaccard_sparse3D_decoder.csv'))
    decoder = decoder[(['train', 'validation', 'Mode', 'Total_epochs'])]
    finetune = pd.read_csv(os.path.join(PATH, folder_name + '_{}'.format(i), '1_jaccard_sparse3D_decoder.csv'))
    finetune = finetune[(['train', 'validation', 'Mode', 'Total_epochs'])]
    last_epoch_decoder = np.max(decoder['Total_epochs'])
    finetune['Total_epochs'] = finetune['Total_epochs'] + last_epoch_decoder
    aux = pd.concat([decoder, finetune])
    aux.to_csv(os.path.join(PATH, 'experiments_logs', '1_jaccard_sparse3D_' + folder_name + '_{}.csv'.format(i)))

###


path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/mobilenet_mobileunet_lstm_tips_large_onlydecoder_v02/epoch_loss.csv"
data = pd.read_csv(path)
fig = plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
val = smooth(data.validation, 100)
plt.plot(np.asarray(data.Epochs.iloc[:-50]), np.asarray(data.validation.iloc[:-50]), alpha=0.5, color="coral")
plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color="coral")
val = smooth(data.train, 100)
plt.plot(np.asarray(data.Epochs.iloc[:-50]), np.asarray(data.train.iloc[:-50]), alpha=0.5, color="lightseagreen")
plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color="lightseagreen")
plt.xlabel('Epochs', fontsize=12)
# plt.ylim([0, 0.2])
plt.xlim([0, 4000])
plt.legend(["Validation", "", "Train", ""])
plt.ylabel('Loss function', fontsize=12)
# plt.legend(legend, loc='upper right', bbox_to_anchor=(1.12, 1.0), fontsize=12)

path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/mobilenet_mobileunet_lstm_tips_large_onlydecoder_v02/1_jaccard_sparse3D.csv"
data = pd.read_csv(path)
plt.subplot(2, 1, 2)
val = smooth(data.validation, 100)
plt.plot(np.asarray(data.Epochs.iloc[:-50]), np.asarray(data.validation.iloc[:-50]), alpha=0.5, color="coral")
plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color="coral")
val = smooth(data.train, 100)
plt.plot(np.asarray(data.Epochs.iloc[:-50]), np.asarray(data.train.iloc[:-50]), alpha=0.5, color="lightseagreen")
plt.plot(np.asarray(data.Epochs.iloc[:-50]), val[:-50], alpha=1, color="lightseagreen")

plt.ylabel('Jaccard index', fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.xlim([0, 4000])
plt.ylim([0, 0.8])
plt.tight_layout()
fig.savefig(os.path.join(output_path, 'large_decoder.pdf'), format='pdf', dpi=500, transparence=True)
fig.savefig(os.path.join(output_path, 'large_decoder.png'), format='png', dpi=500)
plt.show()

###

output_path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/plots"
path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/epoch_loss_large_v01_total.csv"
data = pd.read_csv(path)
fig = plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
val = smooth(data.validation, 100)
data["validation_smooth"] = val
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="validation", hue="Mode", alpha=0.5, palette="hls")
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="validation_smooth", hue="Mode", palette="hls")
plt.legend(["Validation", ""])
# plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color=colors[3])
val = smooth(data.train, 100)
data["train_smooth"] = val
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="train", hue="Mode", linestyle='dashed', alpha=0.5)
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="train_smooth", hue="Mode", linestyle='dashed')
plt.legend([], [], frameon=False)
# plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color=colors[3], linestyle='dashed')
plt.xlabel('Epochs', fontsize=12)
plt.ylim([0, 0.2])
plt.xlim([0, 19000])
plt.ylabel('Loss function', fontsize=12)
# plt.legend(legend, loc='upper right', bbox_to_anchor=(1.12, 1.0), fontsize=12)

path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/1_jaccard_sparse3D_large_v01_total.csv"
data = pd.read_csv(path)
plt.subplot(2, 1, 2)
val = smooth(data.validation, 100)
data["validation_smooth"] = val
# plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color=colors[3])
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="validation", hue="Mode", alpha=0.5, palette="hls")
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="validation_smooth", hue="Mode", palette="hls")
val = smooth(data.train, 100)
data["train_smooth"] = val
# plt.plot(np.asarray(data.Epochs.iloc[:-50]), val[:-50], alpha=1, color=colors[3], linestyle='dashed')
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="train", hue="Mode", alpha=0.5)
sns.lineplot(data=data.iloc[:-50], x="Total_epochs", y="train_smooth", hue="Mode")
plt.legend(["Validation", ""])
plt.ylabel('Jaccard index', fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.xlim([0, 19000])
plt.ylim([0, 1])
# plt.legend([], [], frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(output_path, 'bt2_final.pdf'), format='pdf', dpi=500, transparence=True)
fig.savefig(os.path.join(output_path, 'bt2_final.png'), format='png', dpi=500)
plt.show()




###

colors = ["#2414FF", "#FF2424", "#4FA13D", "#DA2BED", "#FF830F"]
alpha = 0.1
PATH = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/experiments_logs"
output_path = "/home/esgomezm/Documents/3D-PROTUCEL/MU-Lux-CZ/SERVER_RESULTS/plots"
folder_name = 'mobilenet_lstm_experiment'
# folder_name = 'mobilenet_lstm_masks_experiment'
name=[]
fig = plt.figure(figsize=(20, 10))
for i in range(1, 6):
    loss = pd.read_csv(os.path.join(PATH, 'epoch_loss_' + folder_name + '_{}.csv'.format(i)))
    plt.subplot(3, 1, 1)
    val = np.array(loss.validation)
    val = smooth(val, 100)
    loss["validation_smooth"] = val
    plt.plot(loss["Total_epochs"].iloc[:-50], loss["validation"].iloc[:-50], alpha=alpha, color=colors[i-1])
    plt.plot(loss["Total_epochs"].iloc[:-50], loss["validation_smooth"].iloc[:-50], color=colors[i-1])
    name.append('BT{}'.format(i))
    # name.append('B{}'.format(i))
    name.append('')

    plt.subplot(3, 1, 2)
    val = np.array(loss.train)
    val = smooth(val, 100)
    loss["train_smooth"] = val
    plt.plot(loss["Total_epochs"].iloc[:-50], loss["train"].iloc[:-50], alpha=alpha, color=colors[i-1], linestyle='dashed')
    plt.plot(loss["Total_epochs"].iloc[:-50], loss["train_smooth"].iloc[:-50], color=colors[i-1], linestyle='dashed')


    jaccard = pd.read_csv(os.path.join(PATH, '1_jaccard_sparse3D_' + folder_name + '_{}.csv'.format(i)))
    plt.subplot(3, 1, 3)
    val = np.array(jaccard.validation)
    val = smooth(val, 100)
    jaccard["validation_smooth"] = val
    # plt.plot(data.Epochs.iloc[:-50], val[:-50], alpha=1, color=colors[3])
    plt.plot(jaccard["Total_epochs"].iloc[:-50], jaccard["validation"].iloc[:-50], alpha=alpha, color=colors[i-1])
    plt.plot(jaccard["Total_epochs"].iloc[:-50], jaccard["validation_smooth"].iloc[:-50], color=colors[i-1])
    val = np.array(jaccard.train)
    val = smooth(val, 100)
    jaccard["train_smooth"] = val
    plt.plot(jaccard["Total_epochs"].iloc[:-50], jaccard["train"].iloc[:-50], alpha=alpha, color=colors[i-1], linestyle='dashed')
    plt.plot(jaccard["Total_epochs"].iloc[:-50], jaccard["train_smooth"].iloc[:-50], color=colors[i-1], linestyle='dashed')

plt.subplot(3, 1, 1)
plt.xlabel('Epochs', fontsize=12)
plt.ylim([0, 0.18])
plt.xlim([0, 4000])
plt.ylabel('Loss function (validation)', fontsize=12)
plt.legend(name, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=12)

plt.subplot(3, 1, 2)
plt.xlabel('Epochs', fontsize=12)
plt.ylim([0, 0.18])
plt.xlim([0, 4000])
plt.ylabel('Loss function (train)', fontsize=12)

plt.subplot(3, 1, 3)
plt.ylabel('Jaccard index', fontsize=12)
plt.xlabel('Epochs', fontsize=12)
plt.xlim([0, 4000])
plt.ylim([0, 1])
plt.legend(['', 'Validation', '', 'Train'], frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(output_path, folder_name + '.pdf'), format='pdf', dpi=500, transparence=True)
fig.savefig(os.path.join(output_path, folder_name + '.png'), format='png', dpi=500)
plt.show()


