# PAMIL
## PAMIL: Prototype Attention-based Multiple Instance Learning for Whole Slide Image Classification
MICCAI 2024

![Overview of PAMIL](.\.github\overview.png)

**Abstract.** Digital pathology images are not only crucial for diagnosing cancer but also play a significant role in treatment planning, and research into disease mechanisms. The multiple instance learning (MIL) technique provides an effective weakly-supervised methodology for analyzing gigapixel Whole Slide Image~(WSI). Recent advancements in MIL approaches have predominantly focused on predicting a singular diagnostic label for each WSI, simultaneously enhancing interpretability via attention mechanisms.
However, given the heterogeneity of tumors, each WSI may contain multiple histotypes. Also, the generated attention maps often fail to offer a comprehensible explanation of the underlying reasoning process.
These constraints limit the potential applicability of MIL-based methods in clinical settings. In this paper, we propose a Prototype Attention-based Multiple Instance Learning (PAMIL) method, designed to improve the model's reasoning interpretability without compromising its classification performance at the WSI level. PAMIL merges prototype learning with attention mechanisms, enabling the model to quantify the similarity between prototypes and instances, thereby providing the interpretability at instance level. Specifically, two branches are equipped in PAMIL, providing prototype and instance-level attention scores, which are aggregated to derive bag-level predictions. Extensive experiments are conducted on four datasets with two diverse WSI classification tasks, demonstrating the effectiveness and interpretability of our PAMIL.

## Training Model

Depending on the task, we provide the project code for the multi-class and multi-label tasks.

### Preparing Dataset

Before training the model, ensure that you save your patch features in the `--data_root_dir`. There is no limitation on obtaining patch features. In this paper, pre-trained ResNet50 is directly used to extract PATCH features. Store the features in an `.npy` file, using a dictionary with the key `feature` to save the features.

Next, save the dataset labels in separate CSV files to distinguish between different datasets. In each CSV file, `case_id` should represent the MD5 hash of the sample, and `slide_id` should represent the TCGA code of the slide. For multi-class tasks, `label` should be the unique label of the sample. For multi-label tasks, `label` should be a string of binary values like '0,1,0' indicating the presence or absence of each class.

### Initialization of the Prototypes

After the data is organized, set the hyperparameters of `clustering.py` according to your set, and you can get your initial prototypes.

### Running Main Function

Finally, you can train the model using the main function. It is worth noting that you need to modify the input parameters according to the actual situation. 
