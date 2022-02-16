import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

def difference(start, end):
    return ((end-start)/start)*100

input_list = [
	"video/240p_60fps.mp4",
	"video/360p_30fps.mp4",
	"video/480p_30fps.mp4",
	"video/720p_30fps.mp4",
	"video/1080p_30fps.mp4",
	"video/1080p_60fps.mp4",
	"csi://0", 
	"/dev/video1"
			]


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

    size_ssd_dist = 3.5
    size_student_ssd_dist = 0.9

    diff = difference(size_ssd, size_ssd_dist)

    plt.rcParams["figure.figsize"] = [5.50, 2.5]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    plt.xlabel('MByte')
    ax.barh('Distilled', [size_ssd_dist], height=.5, color="red", align='center')
    b5 = ax.barh('Distilled', [size_student_ssd_dist], height=.5, color="green", align='center')
    b2 = ax.barh('Original', [size_ssd], height=.5, color="red", align='center')
    b3 = ax.barh('Original', [size_teacher_SSD], height=.5, color="blue", align='center')
    ax.text(size_ssd_dist, -0.1, str(format(diff, '.2f'))+'%', color='black', fontweight='bold', fontsize = 14)

    plt.legend([b2, b3, b5], ["SSD", "Teacher", "Student"], title="Models", loc="lower right")
    plt.title("Difference size before Original SSD and Distilled SSD")

    #plt.show()
    plt.savefig('./images/Different size SSD' + '.png')

#plot_diff_SSD_size()

def plot_diff_teacher_stud():
    size_teach = 13
    size_stud = 0.948

    diff = difference(size_teach, size_stud)

    plt.rcParams["figure.figsize"] = [5.50, 2.5]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    plt.xlabel('MByte')
    b1 = ax.barh('Distilled', [size_stud], height=.5, color="green", align='center')
    b2 = ax.barh('Original', [size_teach], height=.5, color="red", align='center')

    ax.text(size_stud, -0.1, str(format(diff, '.2f'))+'%', color='black', fontweight='bold', fontsize=14)

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

    ax.text(n2, -0.1, str(format(percentage, '.2f'))+"%", color='black', fontweight='bold', fontsize=14)
    if is_legend:
        plt.legend([b2, b1], ["Teacher", "Student", ], title="Models", loc="lower right")

    #plt.show()
    plt.savefig('./images/'+title + '.png')


# #Diff param base_net
# par_base_net_orig = 3215176
# par_base_net_dist = 215128
# title = "Difference base_nets parameters"
# plot_diff_params(par_base_net_orig, par_base_net_dist, difference(par_base_net_orig,par_base_net_dist), title)

# par_ssd_orig = 7643796
# par_ssd_dist = 861828
# title = "Difference SSD parameters"
# plot_diff_params(par_ssd_orig, par_ssd_dist, difference(par_ssd_orig, par_ssd_dist), title, False)   

def diff_FPS(path_distill, path_original, name):
    #path_distill = '/home/flavio/thesis/jetson_nano/jetson_benchmarks/benchmarks_pt2/obj_detection_ssd_distill/ssd_std_dst_obj_detection_Freeze_mean.csv'
    #path_original = '/home/flavio/thesis/jetson_nano/jetson_benchmarks/benchmarks_pt2/obj_detection_ssd_distill/ssd_teacher_object_detection_mean.csv.csv'
    dataframe_distill = pd.read_csv(path_distill, index_col=[0])
    dataframe_original = pd.read_csv(path_original, index_col=[0])
    width = 0.45
    plt.rcParams["figure.figsize"] = [8, 4]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()

    dict_out = {
        'Dist': [],
        'Original': [],
        'Percentage': []
    }
    dict_net = {
        'Dist': [],
        'Original': [],
        'Percentage': []
    }

    lst_source = ['W1', 'W2', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for col in dataframe_distill.columns:
        if col != 'Input':
            value_end = dataframe_distill[col].values
            value_start = dataframe_original[col].values
            percentage = [difference(value_start[x], value_end[x]) for x in range(0,len(lst_source))]
            if col == "Output":
                dict_out['Dist'] = value_end
                dict_out['Original'] = value_start
                dict_out['Percentage'] = percentage
            elif col == 'Video':
                dict_net['Dist'] = value_end
                dict_net['Original'] = value_start
                dict_net['Percentage'] = percentage

    ind = np.arange(len(lst_source))
    ax.bar(ind, dict_net['Dist'], width, color="green", align='center')
    ax.bar(ind, dict_net['Original'], width, color="red", align='center')
    rects = ax.patches

    idx = 0
    for rect, label in zip(rects, dict_net['Dist']):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, '+'+str(int(dict_net['Percentage'][idx])) + "%", ha="center", va="bottom", color='green', fontweight='bold', fontsize=12)
        ax.text(rect.get_x() + rect.get_width() / 2, height - 9, str(int(label)), ha="center", va="bottom", color='white', fontweight='bold', fontsize=10)
        ax.text(rect.get_x() + rect.get_width() / 2, int(dict_net['Original'][idx]) - 9, str(int(dict_net['Original'][idx])), ha="center", va="bottom", color='white', fontweight='bold', fontsize=10)
        idx += 1

    plt.ylabel('FPS')
    plt.title('Jetson: Mean FPS difference between Original SSD and Distilled SSD (TensorRT)')
    plt.xticks(ind, ('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'W1', 'W2'))
    plt.yticks(np.arange(0, max(dict_net['Dist'])+20, 10))
    #plt.show()

    plt.savefig(name+'.png')

path_distill = '/home/flavio/thesis/jetson_nano/jetson_benchmarks/benchmarks_pt2/SSD_Distill_obj_detection/MAXP_ssd_studet_Distilled_object_detection_mean.csv'
path_original = '/home/flavio/thesis/jetson_nano/jetson_benchmarks/benchmarks_pt2/SSD_Distill_obj_detection/MAXP_ssd_teacher_object_detection_mean.csv'
diff_FPS(path_distill, path_original, './images/Mean Difference FPS Jetson TensorRT')