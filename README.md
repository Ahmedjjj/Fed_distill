# dataset-distillation


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
The code is under `fed_distill`, and is fairly well documented.

# Reproducibility of experiments
For ease of training and reproducibility, the experiments were run in a pipeline structure, with the following stages:
1. Data splitting according to a given strategy. This corresponds to the file `data_split.py`
2. Training of teachers. This coresponds to to the file `train_teachers.py` 
3. Generating initial batches using deep inversion: This corresponds to the file `generate_initial.py`
4. Finally, student training in the file `train_student.py`

The code is run with the amazing [Hydra](https://hydra.cc/) library. Please take a look at the [documentation](https://hydra.cc/docs/intro/) for more info. 
## Experiments from Section III (Reproducing Deep Inversion)

```bash
# No splitting is needed (1 teacher)

# Train teacher
python train_teachers.py +teacher=full_1_teacher \
                          teacher.save_folder=(ADD HERE)

# Generate initial batches
python generate_initial.py +student=1_teacher_custom \
                            teacher.save_folder=(ADD Here (same as before)) \
                            initial.save_path=(ADD HERE)

# Train without adaptiveness
python train_student.py +student=1_teacher_custom \
                         teacher.save_folder=(ADD Here (same as before)) \
                         initial.save_path=(ADD Here (same as before)) \
                         student.save_folder=(ADD HERE)

# Train with adaptiveness
python train_student.py +student=1_teacher_custom_adap \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD HERE)

# Train non adaptive with the same code as adaptive while setting the competition scale to 0
python train_student.py +student=1_teacher_custom_adap \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD HERE) \
                        deep_inv.adi.loss.comp_scale=0

# Train with parameters from the paper
python generate_initial.py +student=1_teacher_paper \
                            teacher.save_folder=(ADD Here (same as before)) \
                            initial.save_path=(ADD HERE)
python train_student.py +student=1_teacher_paper  \
                        teacher.save_folder=(ADD Here (same as before)) \
                        initial.save_path=(ADD Here (same as before)) \
                        student.save_folder=(ADD HERE)
```

### Experiments from Section IV (Federated Learning with dataset distillation)
## 1 teacher with half the data at the student
```bash
# split the data
python data_split.py +split=full_2_teachers split.save_path=(ADD Here)

# train the teachers
p

```

_
# Acknowledgments
