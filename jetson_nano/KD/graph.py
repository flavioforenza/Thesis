import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_accuracy():
    file = open("./results/all_accuracy_temps.pkl", "rb")
    acc_temps = pickle.load(file)

    labels = ['Teacher', 'Student', 'T=1', 'T=2', 'T=3', 'T=4', 'T=5', 'T=10', 'T=15']

    top5_acc = [float(x[1]) for x in acc_temps.values()]
    top1_acc = [float(x[0]) for x in acc_temps.values()]


    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, top1_acc, width, label='Top1')
    rects2 = ax.bar(x + width/2, top5_acc, width, label='Top5')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy at Top1 and Top5')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    #plt.show()

    plt.savefig('images/Top1 and Top5 Accuracy'+'.png')

#plot_accuracy()


def plot_diff_SSD_size():
    size_ssd = 30.7
    size_teacher_SSD = 13

    size_ssd_dist = 2.6
    size_student_ssd_dist = 0.9

    plt.rcParams["figure.figsize"] = [5.50, 2.5]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    plt.xlabel('MByte')
    ax.barh('Distilled', [size_ssd_dist], height=.5, color="red", align='center')
    b5 = ax.barh('Distilled', [size_student_ssd_dist], height=.5, color="green", align='center')
    b2 = ax.barh('Original', [size_ssd], height=.5, color="red", align='center')
    b3 = ax.barh('Original', [size_teacher_SSD], height=.5, color="blue", align='center')
    ax.text(3, -0.1, '-88,60%', color='black', fontweight='bold', fontsize = 14)

    plt.legend([b2, b3, b5], ["SSD", "Teacher", "Student"], title="Models", loc="lower right")
    plt.title("Difference size before Original SSD and Distilled SSD")

    #plt.show()
    plt.savefig('./images/Different size SSD' + '.png')

#plot_diff_SSD_size()

def plot_diff_teacher_stud():
    size_teach = 13
    size_stud = 0.9

    plt.rcParams["figure.figsize"] = [5.50, 2.5]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    plt.xlabel('MByte')
    b1 = ax.barh('Distilled', [size_stud], height=.5, color="green", align='center')
    b2 = ax.barh('Original', [size_teach], height=.5, color="red", align='center')

    ax.text(1, -0.1, '-92.71%', color='black', fontweight='bold', fontsize=14)

    plt.title("Difference size before teacher and student")

    plt.legend([b2, b1], ["Teacher", "Student",], title="Models", loc="lower right")

    #plt.show()
    plt.savefig('./images/Different size base_net' + '.png')

#plot_diff_teacher_stud()

def plot_diff_params(n1, n2,percentage, title, is_legend=True):

    plt.rcParams["figure.figsize"] = [5.50, 2.5]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    plt.title(title)

    plt.xlabel('Million Prameters')
    b1=ax.barh('Distilled', [n2], height=.5, color="green", align='center')
    b2=ax.barh('Original', [n1], height=.5, color="red", align='center')

    ax.text(n2, -0.1, "-"+str(percentage)+"%", color='black', fontweight='bold', fontsize=14)
    if is_legend:
        plt.legend([b2, b1], ["Teacher", "Student", ], title="Models", loc="lower right")

    #plt.show()
    plt.savefig('./images/'+title + '.png')


#Diff param base_net
par_base_net_orig = 3215176
par_base_net_dist = 215128
title = "Difference base_nets parameters"
plot_diff_params(par_base_net_orig, par_base_net_dist, 93.3, title)

par_ssd_orig = 7643796
par_ssd_dist = 861828
title = "Difference SSD parameters"
plot_diff_params(par_ssd_orig, par_ssd_dist, 88.73, title, False)
