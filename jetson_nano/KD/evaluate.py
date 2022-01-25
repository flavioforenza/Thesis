import matplotlib.pyplot as plt
import numpy as np
import models
import torch.utils.data
import torchvision
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from models import MobileNetV1_Teach, MobileNetV1_Stud

CHECKPOINTS_DIR = './checkpoints'
MODEL_DIR = './model'

def evaluate(model_type='student', prefix='student'):
    if model_type == 'teacher':
        model = MobileNetV1_Teach(8)
    elif model_type == 'student':
        model = MobileNetV1_Stud(8)
       
    model_paths = []
    for epoch_count in range(0, 301, 1):
        model_paths.append(os.path.join(CHECKPOINTS_DIR, '{}-{}.pth'.format(prefix, epoch_count+1)))
    #model_paths.append(os.path.join(MODEL_DIR, '{}.pth'.format(prefix)))
    
    valdir = os.path.join('./data/', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    batch_size = 8

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
error_counts_student = evaluate('student', 'student')
# error_counts_student_distill = evaluate('student', 'student-distill')
# error_counts_teacher_distill = evaluate('teacher', 'teacher-distill')

# Store as file
# np.save('./data/error_counts_teacher.npy', error_counts_teacher)
np.save('./results/error_counts_student.npy', error_counts_student)
# np.save('./data/error_counts_student_distill.npy', error_counts_student_distill)
# np.save('./data/error_counts_teacher_distill.npy', error_counts_teacher_distill)

# # Load from file
#error_counts_teacher = np.load('./data/error_counts_teacher.npy')
error_counts_student = np.load('./results/error_counts_student.npy')
#error_counts_student_distill = np.load('./data/error_counts_student_distill.npy')
#error_counts_teacher_distill = np.load('./data/error_counts_teacher_distill.npy')

# Prepare to plot
fig, ax = plt.subplots()

# Plot error
#ax.plot(range(0, 3001, 100), error_counts_teacher, label='teacher')
ax.plot(range(0, 301, 1), error_counts_student, label='student')
#ax.plot(range(0, 3001, 100), error_counts_student_distill, label='student with distillation')
#ax.plot(range(0, 3001, 100), error_counts_teacher_distill, label='teacher with distillation')
ax.set_xlabel('number of epochs')
ax.set_ylabel('number of errors')
ax.set_title('Learning curve')
ax.legend()

# Show
#plt.show()
plt.savefig('learning_curve.png')