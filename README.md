# Dataset Distillation for One-Shot Federated Learning


This repository contains the code for a project done at the MLO Lab, EPFL, Switzerland.
The title of the project is: *Dataset Distillation for One-Shot Federated Learning*

Please read the report (`report.pdf`) for more details.

# Requirements
The file `requirements.txt` contains the necessary libraries to run the code, along with the currently used versions.  
We use python `3.9.7`.  
You may install the necessay libraries using: 
```bash
pip install -e requirements.txt
```
However, this may not be a good idea, since pytorch may need different versions depending on your cuda version. A good way to simplify this process is to use [light-the-torch](https://github.com/pmeier/light-the-torch).

# Repository structure
The code is under `fed_distill`, and is fairly well documented:
- `cifar10`: contains helpers for loading/visualizing cifar10 images.
- `data`: contains abstractions relating to dataset splitting, `torch` Datasets and DataLoader for datasets that evolove during execution.
- `deep_inv`: contains the code for generating images.
- `resnet`: contains the `ResNet` implementation.
- `train`: contains the code used for training.


# Reproducibility of experiments
For ease of training and reproducibility, the experiments were run in a pipeline structure, with the following stages:
1. Data splitting according to a given strategy. This corresponds to the file `data_split.py`. This will output a `json` file containing the train and test split for each student.
3. Training of teachers. This coresponds to to the file `train_teachers.py`. This will output for each teacher i:
      - `metrics_teacher{i}.pt`, which contains the training metrics for the corresponding teacher.
      - `model_teacher{i}.pt`, which will contain the weights of the best performing model (in terms of test accuracy).
5. Generating initial batches using deep inversion: This corresponds to the file `generate_initial.py`. This will output a file containing a `dict` of two `tensors`, the images and corresponding labels.
7. Finally, student training in the file `train_student.py`. This will output:
      - `dataset.pt`, which contains the full synthetic dataset.
      - `metrics.pt`, which contains the training metrics for the student.
      - `student.pt`, which contains the final student model weights.
      - for each teacher i:
        - `di_metrics_teacher{i}.pt`, which contains the instanteneous accuracy for teacher i and the the student.
 
The files `experimental_sampling_generate_initial.py` and `experimental_sampling_train_student.py` contain the main scripts for training using Sampling Deep Inversion` (Algorithm 1 in the report).

The code is run with the amazing [Hydra](https://hydra.cc/) library. Please take a look at the [documentation](https://hydra.cc/docs/intro/) for more info. 
In the following are the commands to reproduce the experiments from the report. Note that parts with the form (ADD Here) can be modified directly in the config files for ease of use.

## Experiments from Section III (Reproducing Deep Inversion)

```bash
# No splitting is needed (1 teacher)

# train teacher
python train_teachers.py +teacher=full_1_teacher \
                          teacher.save_folder=(ADD HERE)

# generate initial batches
python generate_initial.py +student=1_teacher_custom \
                            teacher.save_folder=(ADD Here (same as before)) \
                            initial.save_path=(ADD HERE)

# train without adaptiveness
python train_student.py +student=1_teacher_custom \
                         teacher.save_folder=(ADD Here (same as before)) \
                         initial.save_path=(ADD Here (same as before)) \
                         student.save_folder=(ADD HERE)

# train with adaptiveness
python train_student.py +student=1_teacher_custom_adap \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD HERE)

# train non adaptive with the same code as adaptive while setting the competition scale to 0
python train_student.py +student=1_teacher_custom_adap \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD HERE) \
                        deep_inv.adi.loss.comp_scale=0

# train with parameters from the paper
python generate_initial.py +student=1_teacher_paper \
                            teacher.save_folder=(ADD Here (same as before)) \
                            initial.save_path=(ADD HERE)
python train_student.py +student=1_teacher_paper  \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD HERE)
```

## Experiments from Section IV (Federated Learning with dataset distillation)
### 1 teacher with half the data at the student
```bash
# split the data
python data_split.py +split=full_2_teachers \
                      split.save_path=(ADD Here)

# train the teachers
python train_teachers.py +teacher=full_2_teachers \
                         +split=full_2_teachers \
                          teacher.save_folder=(ADD Here) \
                          split.save_file=(ADD Here (same as before)
                          
# Generate initial batches
python generate_initial.py +student=half_1_teacher_custom_adap \
                            teacher.save_folder=(ADD Here (same as before)) \
                            split.save_file=(ADD Here (same as before) \
                            initial.save_path=(ADD Here)
# train student 
python train_student.py +student=half_1_teacher_custom_adap \
                         teacher.save_folder=(ADD Here (same as before)) \
                         split.save_file=(ADD Here (same as before) \
                         initial.save_path=(ADD Here (same as before))
                         student.save_folder=(ADD Here)
# train with full data
python generate_initial.py +student=half_1_teacher_custom_adap \
                            teacher.save_folder=(ADD Here (same as before)) \
                            split.save_file=(ADD Here (same as before) \
                            initial.save_path=(ADD Here) \
                            initial.num_batches=50
python train_student.py +student=half_1_teacher_custom_adap \
                         teacher.save_folder=(ADD Here (same as before)) \
                         split.save_file=(ADD Here (same as before) \
                         initial.save_path=(ADD Here (same as before))
                         student.save_folder=(ADD Here) \
                         student.new_batches_per_epoch=2

# weighted training (with a weight of 10)
python train_student.py +student=half_1_teacher_custom_weighted \
                         teacher.save_folder=(ADD Here (same as before)) \
                         split.save_file=(ADD Here (same as before) \
                         initial.save_path=(ADD Here (same as before))
                         student.save_folder=(ADD Here) \
     
```
If you would like to train with a weight of 2 as well, please change `config/student/half_1_teacher_custom_weighted` at line `24` by modifying the weight vector (10.0 becomes 2.0)

### IID 10 teachers
```bash
# split the data
python split_data.py +split=iid_10_teachers \
                     split.save_path=(ADD Here)

# train teachers
python train_teachers.py +teacher=iid_10_teachers \
                      +split=iid_10_teachers \ 
                      split.save_path=(ADD Here (same as before)) \
                      teacher.save_folder=(ADD Here)

# generate initial batches
python generate_initial.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here)

# train student 
python train_student.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here)

# train student with less noisy images (3000 updates per batch)
python generate_initial.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here) \
                        deep_inv.di.grad_updates_batch=3000
 python train_student.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here) \
                        deep_inv.adi.grad_updates_batch=3000
                        
 # train student with sampling (3 teachers) (algorithm 1)
 python experimental_sampling_generate_initial.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here) \
                        deep_inv=sampling_experimental
 python experimental_sampling_train_student.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here) \
                        deep_inv=sampling_experimental
   
 # train student with sampling (5 teachers) (algorithm 1)
 python experimental_sampling_generate_initial.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here) \
                        deep_inv=sampling_experimental \
                        deep_inv.di.num_teachers=5
                        
 python experimental_sampling_train_student.py +student=iid_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_folder=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here) \
                        deep_inv=sampling_experimental \
                        deep_inv.adi.num_teachers=5
```

### Heterogeneous 10 teachers

```bash
# split the data
python split_data.py +split=heter_10_teachers \
                     split.save_path=(ADD Here)

# train teachers
python train_teachers.py +teacher=heter_10_teachers \
                      +split=heter_10_teachers \ 
                      split.save_path=(ADD Here (same as before)) \
                      teacher.save_folder=(ADD Here)

# generate initial batches
python experimental_sampling_generate_initial.py +student=heter_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here) \
                        deep_inv=sampling_experimental

# train student
 python experimental_sampling_train_student.py +student=heter_10_teacher_custom \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here) \
                        deep_inv=sampling_experimental
```

### Fully heterogeneous 2 teachers
```bash
# split the data
python data_split.py +split=full_2_teachers \
                      split.save_path=(ADD Here) \

# train teachers
python train_teachers.py +teacher=full_2_teacher \
                         +split=full_2_teachers \
                          split.save_path=(ADD Here (same as before)) \
                          teacher.save_folder=(ADD Here)
# generate initial batches
python generate_initial.py +student=2_teachers_custom_adap \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here)

# train student 
python train_student.py +student=2_teachers_custom_adap \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here)

# train student with less noisy images (3000 updates per batch)
python generate_initial.py +student=2_teachers_custom_adap \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here) \
                        deep_inv.di.grad_updates_batch=3000
 python train_student.py +student=2_teacher_custom_adap \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here) \
                        deep_inv.adi.grad_updates_batch=3000 
```

### Fully heterogeneous 5 teachers
```bash
# split the data
python data_split.py +split=full_5_teachers \
                      split.save_path=(ADD Here) \

# train teachers
python train_teachers.py +teacher=full_5_teacher \
                         +split=full_2_teachers \
                          split.save_path=(ADD Here (same as before)) \
                          teacher.save_folder=(ADD Here)
# generate initial batches
python generate_initial.py +student=5_teachers_custom_adap \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here)

# train student 
python train_student.py +student=5_teachers_custom_adap \
                        split.save_path=(ADD Here (same as before)) \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD Here)

```
# Acknowledgments
The code for deep inversion is based on the [original paper implementation](https://github.com/NVlabs/DeepInversion).  
The resnet implementation is taken from [this implementation](https://github.com/huawei-noah/Efficient-Computing/blob/master/Data-Efficient-Model-Compression/DAFL/resnet.py).  
We use [Hydra](https://hydra.cc/) for configuring the experiments.
