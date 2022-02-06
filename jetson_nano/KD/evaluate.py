import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import MobileNetV1_Teach, MobileNetV1_Stud

CHECKPOINTS_DIR = './checkpoints'
MODEL_DIR = './model'

resolution=224
alpha=0.25

def evaluate(model_type='student', prefix='student'):
    if model_type == 'teacher':
        model = MobileNetV1_Teach(8)
    elif model_type == 'student':
        model = MobileNetV1_Stud(8, alpha)
       
    model_paths = []
    for epoch_count in range(0, 1000, 1):
        model_paths.append(os.path.join(CHECKPOINTS_DIR, '{}-{}.pth'.format(prefix, epoch_count+1)))
    #model_paths.append(os.path.join(MODEL_DIR, '{}.pth'.format(prefix)))
    
    valdir = os.path.join('./data/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    batch_size = 8

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    error_counts = np.zeros(shape=len(model_paths), dtype=np.int32)
    for model_count, model_path in enumerate(model_paths):
        md = torch.load(model_path)
        model.load_state_dict(md.state_dict())
        model.eval()
        model = model.cuda()

        error_count = 0
        gpu=0
        for _, (images, target) in enumerate(data_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)
            # compute output
            output = model(images)
            #loss = criterion(output, target)
            with torch.no_grad():
                output = torch.argmax(output, dim=1)
                error_count += batch_size - torch.eq(output, target).sum()
        
        error_counts[model_count] = error_count
        print('{}-{}: {}'.format(prefix, model_count, error_count))

    return error_counts

# Evaluate accuracy
#error_counts_teacher = evaluate('teacher', 'teacher')
#error_counts_student = evaluate('student', 'student_distill')
# error_counts_student_distill = evaluate('student', 'student-distill')
# error_counts_teacher_distill = evaluate('teacher', 'teacher-distill')

# Store as file
#np.save('./results/error_counts_teacher_normal2.npy', error_counts_teacher)
#np.save('./results/error_counts_student_distill__0.25_224_t=2.npy', error_counts_student)
# np.save('./results/error_counts_student_distill.npy', error_counts_student_distill)
# np.save('./results/error_counts_teacher_distill.npy', error_counts_teacher_distill)

# # Load from file
error_counts_teacher = np.load('./results/final_error_counts_teacher_normal.npy')
#error_counts_student = np.load('./results/error_counts_student_conv2D_half.npy')
#error_counts_student_2 = np.load('./results/error_counts_student_conv2D_half_NoBatch.npy')
error_counts_student_3 = np.load('./results/error_counts_student_0.25_224.npy')
error_counts_student_4 = np.load('./results/error_counts_student_0.25_128.npy')
error_counts_student_distill_0_24_224_t1 = np.load('./results/error_counts_student_distill__0.25_224_t=1.npy')
error_counts_student_distill_0_24_224_t2 = np.load('./results/error_counts_student_distill__0.25_224_t=2.npy')
error_counts_student_distill_0_24_224_t3 = np.load('./results/error_counts_student_distill__0.25_224_t=3.npy')
error_counts_student_distill_0_24_224_t5 = np.load('./results/error_counts_student_distill__0.25_224_t=5.npy')
error_counts_student_distill_0_24_224_t10 = np.load('./results/error_counts_student_distill__0.25_224_t=10.npy')
error_counts_student_distill_0_24_224_t15 = np.load('./results/error_counts_student_distill__0.25_224_t=15.npy')

#error_counts_teacher_distill = np.load('./results/error_counts_teacher_distill.npy')

print("Minimo errore: ", min(error_counts_student_distill_0_24_224_t3))
print("Massimo errore: ", max(error_counts_student_distill_0_24_224_t3))

# Prepare to plot
fig, ax = plt.subplots()

# Plot error
#ax.plot(range(0, 3001, 100), error_counts_teacher, label='teacher')
#0,1000,99
#print(len(error_counts_teacher))
lst_y_teach = [error_counts_teacher[x-1] for x in range(1,1001, 249)]
print(len(lst_y_teach))

#lst_y_stud = [error_counts_student[x-1] for x in range(1,1001, 199)]
#lst_y_stud_2 = [error_counts_student_2[x-1] for x in range(1,1001, 199)]
lst_y_stud_3 = [error_counts_student_3[x-1] for x in range(1,1001, 249)]
lst_y_stud_4 = [error_counts_student_4[x-1] for x in range(1,1001, 249)]
lst_y_stud_5 = [error_counts_student_distill_0_24_224_t1[x-1] for x in range(1,1001, 249)]
lst_y_stud_6 = [error_counts_student_distill_0_24_224_t5[x-1] for x in range(1,1001, 249)]
lst_y_stud_7 = [error_counts_student_distill_0_24_224_t10[x-1] for x in range(1,1001, 249)]
lst_y_stud_8 = [error_counts_student_distill_0_24_224_t15[x-1] for x in range(1,1001, 249)]
lst_y_stud_9 = [error_counts_student_distill_0_24_224_t2[x-1] for x in range(1,1001, 249)]
lst_y_stud_10 = [error_counts_student_distill_0_24_224_t3[x-1] for x in range(1,1001, 249)]
print(len(lst_y_stud_10))

#0,1001,100
lst_x = [x for x in range(0,1001, 250)]
print(len(lst_x))

name_img = 'student_distill_0.25_224_T3'

ax.plot(lst_x, lst_y_teach, label='tch_loss')
#ax.plot(lst_x, lst_y_stud, label='std_C2DH_loss')
#ax.plot(lst_x, lst_y_stud_2, label='std_C2DHNB_loss')
ax.plot(lst_x, lst_y_stud_3, label='std_0.25_224loss')
ax.plot(lst_x, lst_y_stud_4, label='std_0.25_128loss')
ax.plot(lst_x, lst_y_stud_5, label='std_D_0.25_224_T1')
ax.plot(lst_x, lst_y_stud_9, label='std_D_0.25_224_T2')
ax.plot(lst_x, lst_y_stud_10, label='std_D_0.25_224_T3')
ax.plot(lst_x, lst_y_stud_6, label='std_D_0.25_224_T5')
ax.plot(lst_x, lst_y_stud_7, label='std_D_0.25_224_T10')
ax.plot(lst_x, lst_y_stud_8, label='std_D_0.25_224_T15')

#ax.plot(range(0, 3001, 100), error_counts_teacher_distill, label='teacher with distillation')

ax.set_xlabel('number of epochs')
ax.set_ylabel('number of errors')
ax.set_title('Learning curve')
ax.legend()

# Show
#plt.show()
plt.savefig(name_img+'.png')