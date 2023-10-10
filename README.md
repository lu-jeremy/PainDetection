# Detecting Nociceptive Intensity Ratings in Patients’ fMRI Scans
Veterans who come back from war cannot fully convey pain reactions due to cognitive disorders, such as Post Traumatic Stress Disorder (PTSD). This work outlines the creation of a machine learning based system to binary-classify pain using a brain imaging dataset. Moreover, the OpenNeuro website conducted an experiment where heat-induced, nociceptive reactions correlated self regulated pain with the ventromedial prefrontal cortex (vmPFC) and nucleus accumbens (NAc) of the brain. Although functional Magnetic Resonance Imaging (fMRI) has been used in said studies for image reconstruction, data analysis, and cognitive behavior simulation, it has not been involved in the task of pain detection at any level. Therefore, the present study proposes a device that uses fMRI images to massage a specific area of the body to help veterans cope with tremendous war injuries. The novel machine learning model predicted pain at an accuracy of 68.49% and is the first to have been created for this task.

## Description

### Data format
* fMRI
  * Pros: Spatial resolution
  * Cons: Lacks temporal resolution (non-temporal state does not allow for immediate processing)
* EEG
  * Pros: Temporal resolution
  * Cons: Lacks spatial resolution (does not show activity in specific regions of the brain)
* fNIRS
  * Spinal cord (experimental, undecided)

### Pre-processing
  - 4D array fMRI scans taken from https://openfmri.org/dataset/
  - dimensionality reduction of 2D data and 3D data
  - large amounts of RAM needed
  - preallocation of numpy arrays
  - considering lazy loading with tf.data
  - pain ratings on a scale from 0-200 are extracted from the dataset
  - binary classification
    - one hot encode labels
    - threshold = 50
    - Verdict: > 50 = pain, < 50 = no pain

### Architecture
  - Convolution2D/3D + AveragePooling + LeakyRelu (alpha = .3) + BatchNormalization + Dropout
  - Dropout used for first block of convolutional layers, tends to drop performance when used in further blocks succeeding the first block
  - Sigmoid activation for binary classification, linear actiation for regression
  
### Training
  - Grid search used for hyperparameter optimization (ran for 36 hours)
    - AveragePooling shown to have a better performance (+ ~10% accuracy) than MaxPooling
    - Lesser values of Dropout (.1, .15)
  - Explainability
    - Saliency maps show significance of pixels relative to the derivative of the weights during backprop (visualization tool: keras-vis)
    - Surrogate models
    - Connectivity correlation for understanding ROI for nociceptive input in the brain
  - RAM exceeds limit during training, possible solutions:
    - swap space w/ virtual ram from SSD
    - AWS or Google Cloud servers
    - remote desktop access
    - recurrently load new data in batches + train on new batches, then save the model and load it back up in a new runtime
  - Do not use LSTM, requires significantly more data to store in Many2Many
  - Accuracy: ~64%
  
### Evaluation
  - When training on a specific portion of the dataset, the model evaluates very poorly (~30% accuracy) on other portions of the dataset
  - Distribution is not equal due to hardware restrictions (memory)

### Prediction
  - shows direct correlation between pain/no pain and pain ratings during prediction process
