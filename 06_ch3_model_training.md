# **Chapter 3.3 – Model Development and Training**

## **3.3.1 Introduction**

The SeeSense project addresses a deceptively simple question: what must a pedestrian who can't see be told about the space immediately in front of them, and how quickly must that information arrives? An assistive navigation aid for visually impaired users is not a general-purpose scene-understanding system. It is a real-time safety instrument whose value collapses if it is slow, if it misses the obstacle that matters, or if it announces obstacles that are not there. The perception component at the center of this system is an object detector, and this chapter documents the complete engineering research process through which that detector was designed, trained, diagnosed, repaired, extended, and ultimately replaced by a stronger successor.

A dedicated detector was required rather than an off-the-shelf pretrained model for reasons that are grounded in the operational domain rather than in novelty. Publicly available detection models are trained predominantly on object vocabularies drawn from general photography and autonomous-driving benchmarks.

Those vocabularies contain people, cars, bicycles and traffic lights, all of which are relevant to pedestrian navigation, but they omit almost everything that constitutes a walking hazard: bollards, curbs, potholes, open manholes, construction barriers, standing bins, staircases, and painted crossings. A model that detects a car with high fidelity but cannot perceive the bollard directly in the user's path is of limited assistive value. The research objective was therefore to construct a detector whose class vocabulary is defined by the navigation task itself, and whose accuracy and latency characteristics are compatible with deployment on constrained hardware.

Three research goals guided the work.

The first was to determine empirically whether a compact detection architecture built from first principles could reach usable accuracy on the target domain, or whether the maturity of established single-stage detection frameworks was a practical prerequisite.

The second was to establish a reliable, reproducible training and evaluation protocol capable of producing trustworthy comparisons between successive model versions.

The third was to expand the range of objects the model could detect, making it better suited to the needs of real-world navigation assistance.

The chapter is written as a continuous engineering narrative rather than as a set of independent experiments. Each stage described below exists because the preceding stage failed in a specific, diagnosable way. The custom detector was abandoned because of measurable architectural limits, not because a newer model was fashionable. The YOLOv8 baseline was superseded because its recall was insufficient for the safety-critical classes. The dataset-expansion phase produced both real gains and an instructive catastrophic failure, and the final YOLO26 stage was designed directly around the lessons those failures produced. Read in sequence, the four stages form a single trajectory from an unevaluable prototype to a validated, exported production model.

## **3.3.2 Research Strategy**

The research methodology followed an iterative diagnose-and-respond cycle. Each experiment was designed not merely to improve a metric but to answer a specific question raised by the failure of the previous experiment. Where an experiment produced an unexpected result, the subsequent experiment was directed at explaining that result rather than at incrementally tuning hyperparameters around it. This discipline is what distinguishes an engineering research process from a hyperparameter search, and it is the reason the final model differs from the first not only in weights but in architecture, dataset, evaluation protocol, and operational infrastructure.

The trajectory proceeded through four stages. The first stage implemented a custom single-stage detector consisting of a pretrained ResNet-18 feature extractor and a hand-written grid-based detection head, together with a custom composite loss function, a manual decoding routine, and a hand-implemented non-maximum suppression procedure.

The purpose of this stage was to establish a controlled baseline in which every component was fully understood and inspectable. Its failure was informative rather than merely disappointing: the model learned to reduce its loss while producing detections that were qualitatively useless, and critically, the experiment contained no mechanism for measuring detection quality in standard terms. That absence of a proper evaluation metric was itself the most important finding of the stage.

The second stage responded directly to those deficiencies by adopting the YOLOv8 framework, which supplies a multi-scale feature pyramid, anchor-free assignment, distribution-focal box regression, integrated augmentation, and decisively, a standard mean-average-precision evaluation pipeline.

This produced the project's first genuine quantitative baseline and, with it, the first genuine diagnosis: precision was acceptable, but recall was poor, and the low recall was concentrated in small, dense, and structurally repetitive classes.

The third stage addressed the data rather than the model. Recognizing that the ten-class vocabulary was insufficient for navigation and that the low recall was partly a data-distribution problem, seven auxiliary single-class datasets were acquired, remapped into the project label space, and used to extend the vocabulary from ten to fourteen classes.

This phase included fine-tuning, targeted oversampling of under-represented classes, and an attempted merge of the original and extended datasets. While some classes achieved their best results up to that point, the experiments also revealed important challenges, including catastrophic forgetting and problems in the dataset merging process.

The fourth and final stage combined every lesson learned. A single unified seventeen-class dataset was assembled and subjected to a formal quality audit before any training was permitted. A YOLO26 model was trained under an explicitly documented, seed-fixed configuration with per-epoch checkpointing to persistent storage and an automatic resume mechanism, evaluated per class, benchmarked for inference latency, and exported to a deployment format.

The evolution can be summarised as Custom Detector → YOLOv8 Baseline → Dataset Expansion, Fine-Tuning and Optimization → YOLO26 Final Model, in which each arrow denotes a specific diagnosed failure rather than a scheduled progression.

## **3.3.3 Stage I – Custom Detector**

### **Objective and motivation**

The first stage aimed to determine whether a compact detector assembled from standard components could achieve usable performance on the SeeSense domain while remaining fully transparent to the developer. Building the detection head, loss function, target encoder, decoder and suppression routine by hand offered complete visibility into the training dynamics and imposed no dependency on an external detection framework. This was a deliberate methodological choice: understanding precisely why a purpose-built detector fails is more instructive than accepting the success of a framework whose internals remain unclear.

### **Architecture**

The model, called "ResNet", combined a torchvision ResNet-18 initialized with ImageNet weights as a frozen feature extractor with a lightweight convolutional detection head.

The backbone included the standard initial convolution, batch normalization, ReLU activation and max-pooling stem followed by the four residual stages, with all backbone parameters frozen so that gradient updates were limited to the head. The head consisted of a 3×3 convolution reducing 512 channels to 256, followed by batch normalization and ReLU, and a 1×1 convolution projecting to the output dimension. With 640×640 input the backbone produced a 20×20 spatial feature map at stride 32, and the head emitted a tensor of shape [B, 20, 20, 2, 15] representing, for each of two predictions per grid cell, one objectness logit, four box parameters, and ten class logits. A forward pass with a batch of two confirmed the expected output shape of torch.Size([2, 20, 20, 2, 15]).

The design followed the original YOLO formulation: a fixed spatial grid, a small, fixed number of predictions per cell, and no anchor priors. A grid-collision analysis routine was implemented to quantify how frequently multiple annotated objects fall within the same grid cell (the condition under which a fixed two-predictions per cell budget silently discards ground-truth objects) which motivated the choice of a 20×20 grid with two boxes per cell as the largest configuration the stride-32 feature map could support.

### **Dataset**

The dataset for this stage consisted of 25,892 training images, 7,399 validation images and 3,699 test images, each with a corresponding YOLO-format annotation file, covering ten classes: person, car, bicycle, motorcycle, bench, fire hydrant, traffic light, stairs, pole and dog. The annotation distribution over the training split was severely imbalanced, as Table 3.1 shows.

**Table 1 — Training annotation counts, ten-class dataset (Stage I and Stage II)**

| **Class**     | **Annotations** |
|---------------|-----------------|
| Pole          | 314,120         |
| Car           | 124,449         |
| Person        | 78,872          |
| Traffic light | 56,978          |
| Bicycle       | 8,752           |
| Motorcycle    | 8,656           |
| Fire hydrant  | 2,740           |
| Bench         | 2,152           |
| Dog           | 2,136           |
| Stairs        | 836             |

The most frequent class was represented by more than three hundred times as many instances as the least frequent. This imbalance would prove to be one of the dominant forces shaping the entire research program, and it reappears in every subsequent stage.

![](media/image1.png)

![](media/image2.png)

### **Training configuration**

Images were loaded through a custom PyTorch Dataset that parsed YOLO-format labels into class-and-box tensors. Training images were augmented with color jitter (brightness, contrast and saturation at 0.2, hue at 0.05) and normalized using ImageNet channel statistics. Validation and test images received normalization only. Bounding boxes were converted to the grid target representation by assigning each box to the cell containing its center and filling the first available of the two prediction slots. Boxes whose cell was already full were discarded.

The composite loss combined binary cross entropy (BCE) with logits for objectness, mean squared error (MSE) for box regression, and cross-entropy for classification, weighted by λ_obj = 1.0, λ_noobj = 0.5, λ_box = 5.0 and λ_cls = 1.0.

Positive and negative objectness terms were averaged separately to prevent the overwhelming majority of empty cells from dominating the gradient. Optimization used Adam at a learning rate of 0.001 applied only to trainable parameters, with a ReduceLROnPlateau scheduler (factor 0.5, patience 1) monitoring validation loss. The batch size was 8 and training was scheduled for ten epochs with early stopping at a patience of three.

An initial loss evaluation on an untrained model returned a total loss of 4.5647, broken down into objectness 1.1239, box 0.1551 and classification 2.6651, consistent with a randomly initialized head over ten classes.

### **Training procedure and results**

Training terminated by early stopping after seven epochs. Table 3.2 records the loss trajectory.

**Table 2 — Custom detector training and validation losses**

| **Epoch** | **Train loss** | **Train obj** | **Train box** | **Train class** | **Val loss** | **Val obj** | **Val box** | **Val class** | **LR**   |
|-----------|----------------|---------------|---------------|-----------------|--------------|-------------|-------------|---------------|----------|
| 1         | 1.3272         | 0.3105        | 0.0417        | 0.8082          | 1.3594       | 0.3548      | 0.0438      | 0.7858        | 0.001000 |
| 2         | 1.2316         | 0.2886        | 0.0403        | 0.7414          | 1.3151       | 0.3519      | 0.0432      | 0.7471        | 0.001000 |
| 3         | 1.1899         | 0.2794        | 0.0398        | 0.7115          | 1.3147       | 0.3582      | 0.0428      | 0.7423        | 0.001000 |
| 4         | 1.1642         | 0.2743        | 0.0395        | 0.6925          | 1.2759       | 0.3452      | 0.0426      | 0.7179        | 0.001000 |
| 5         | 1.1375         | 0.2701        | 0.0393        | 0.6711          | 1.2883       | 0.3470      | 0.0424      | 0.7293        | 0.001000 |
| 6         | 1.1184         | 0.2664        | 0.0391        | 0.6565          | 1.2883       | 0.3356      | 0.0427      | 0.7393        | 0.000500 |
| 7         | 1.0728         | 0.2592        | 0.0387        | 0.6202          | 1.2840       | 0.3485      | 0.0421      | 0.7248        | 0.000500 |

The best validation loss of 1.2759 was recorded at epoch four. The subsequent three epochs produced no improvement and training halted. Training loss continued to fall throughout, from 1.3272 to 1.0728, while validation loss plateaued near 1.28 - the signature of a model beginning to overfit its trainable capacity while failing to generalize.

Qualitative evaluation was more revealing than the loss curves.

Before training, decoding predictions at a confidence threshold of 0.25 yielded zero detections.

After training, decoding at a threshold of 0.55 followed by non-maximum suppression at an IoU (Intersection over Union) threshold of 0.5 produced six surviving detections on a sample image and every one of them was assigned the class "Pole", with confidences between 0.568 and 0.725. Several of the corresponding boxes were degenerate, spanning as little as three pixels in width. Lowering the threshold to 0.50 on another sample produced 54 detections after suppression, indicating that the model's confidence distribution was concentrated in a narrow band where small threshold changes caused order-of-magnitude changes in detection count.

![](media/image3.png)

![](media/image4.png)

### **Failure analysis**

Four distinct failure mechanisms were identified, and each maps directly onto a design decision.

The first was grid capacity saturation. A 20×20 grid with two predictions per cell can represent at most 800 objects per image, but far more restrictively, it can represent at most two objects whose centers fall within the same 32×32-pixel region. In a dataset containing 314,120 pole annotations (frequently appearing as receding rows of near-collinear vertical structures) and 124,449 car annotations in dense traffic scenes, ground-truth objects were systematically discarded during target encoding. The model was therefore trained against targets that were themselves incomplete, and no amount of optimization could recover the discarded objects.

The second was class collapse under extreme imbalance. With poles constituting roughly 51% of all training annotations, the cross-entropy classification term was minimized most efficiently by predicting "Pole" almost everywhere. The observation that every surviving detection carried the pole label is the direct empirical expression of this collapse.

The third was inadequate box regression. Mean squared error on sigmoid-activated width and height values optimizes a quantity only loosely correlated with intersection over union. The box loss fell from 0.0438 to 0.0421 across the entire run, a change of roughly 4% indicating that the localization branch was learning almost nothing. Degenerating three-pixel boxes are the visible consequence.

The fourth, and by far the most consequential, was the absence of a detection metric. The experiment measured loss, and loss alone. There was no mean average precision, no per-class precision or recall, no confusion matrix. Consequently, the run appeared to be succeeding. Training loss fell monotonically while the detector was in fact collapsing to a single class. A pipeline that cannot detect its own failure is not a research instrument.

To these architectural issues must be added two further limitations: the frozen backbone confined all learning to approximately 1.2 million head parameters over features never adapted to the target domain, and the single stride-32 prediction scale left the model structurally incapable of localizing small objects, which constitute the majority of the navigation-relevant vocabulary.

### **Lessons learned and rationale for abandonment**

The architecture was abandoned for reasons that are architectural rather than situational. Repairing would have required implementing multi-scale prediction over a feature pyramid, replacing fixed-slot assignment with anchor-based or anchor-free dynamic label assignment, replacing an IoU-family loss for mean squared error, adding class-balanced sampling or loss weighting, and building a complete mAP evaluation system. Each of these is a component that a mature detection framework already provides in a tested, optimized form. The engineering judgement was that the project's scarce effort was better spent on dataset quality and deployment characteristics than on re-implementing solved problems.

Three durable lessons were carried forward. First, no training run may be considered evaluated in the absence of standard detection metrics. This principle shaped every subsequent stage.

Second, extreme class imbalance is a first-order design constraint rather than an incidental property of the data.

Third, target-encoding schemes must be validated against the actual object density of the dataset before training begins, because information discarded at encoding time is unrecoverable downstream.

## **3.3.4 Stage II – YOLOv8 Baseline**

### **Motivation**

The transition to YOLOv8 followed directly from the four failure mechanisms identified above, and the framework was selected because it resolves each of them structurally. Its feature-pyramid neck with three detection scales addresses the single-scale limitation. Its anchor-free dynamic label assignment eliminates fixed-slot target collisions. Its distribution-focal loss with a complete-IoU objective replaces mean squared error with an objective aligned to localization quality. And its integrated validator produces per-class precision, recall, mAP@50 and mAP@50–95 automatically, eliminating the evaluation blindness that had made the Stage I result uninterpretable. The choice of the nano variant reflected the deployment requirement: the detector must eventually run on constrained hardware, so establishing the baseline at the smallest capacity point yields the most honest picture of what the domain demands.

### **Transfer learning and training configuration**

Training was initialized from the COCO-pretrained "yolov8n.pt" checkpoint. The head was automatically reconfigured from 80 to 10 output classes while the backbone and neck retained their pretrained weights, providing a general-purpose visual feature hierarchy on which the domain-specific head could be trained. The instantiated model comprised 130 layers with 3,012,798 parameters and 8.2 GFLOPs, fusing at inference to 73 layers, 3,007,598 parameters and 8.1 GFLOPs.

Training ran for 100 epochs at 640×640 resolution with a batch size of 16 and an early-stopping patience of 20, on an NVIDIA A100-SXM4-40GB. The optimizer was left in automatic mode, which selected MuSGD at a learning rate of 0.01 with momentum 0.9 across three parameter groups. The default augmentation policy was retained: mosaic augmentation with closure over the final ten epochs, HSV jitter, horizontal flipping, scaling and translation, together with light Albumentations transforms. Each epoch comprised 1,619 iterations, and the complete run required 6.243 hours.

### **Experimental results**

The training trajectory is summarized in Table 3.3.

**Table 3 — YOLOv8n validation metrics during training (selected epochs)**

| **Epoch** | **box_loss** | **cls_loss** | **dfl_loss** | **Precision** | **Recall** | **mAP@50** | **mAP@50–95** |
|-----------|--------------|--------------|--------------|---------------|------------|------------|---------------|
| 1         | 1.774        | 2.022        | 1.241        | 0.507         | 0.295      | 0.300      | 0.177         |
| 3         | 1.874        | 1.671        | 1.314        | 0.426         | 0.240      | 0.226      | 0.119         |
| 10        | 1.716        | 1.390        | 1.234        | 0.580         | 0.338      | 0.362      | 0.209         |
| 20        | 1.643        | 1.286        | 1.191        | 0.627         | 0.372      | 0.409      | 0.247         |
| 30        | 1.607        | 1.235        | 1.173        | 0.654         | 0.390      | 0.431      | 0.263         |
| 50        | 1.560        | 1.177        | 1.148        | 0.648         | 0.404      | 0.443      | 0.274         |
| 70        | 1.522        | 1.127        | 1.126        | 0.653         | 0.407      | 0.449      | 0.280         |
| 90        | 1.472        | 1.068        | 1.102        | 0.668         | 0.408      | 0.455      | 0.286         |
| 100       | 1.444        | 0.987        | 1.079        | 0.668         | 0.411      | 0.458      | 0.289         |

The characteristic transient degradation across epochs two and three, during which mAP@50 fell from 0.300 to 0.226, corresponds to the warm-up phase in which the pretrained head is disrupted before re-converging. Thereafter improvement was monotonic but strongly decelerating: the model reached mAP@50 of 0.431 by epoch 30 and required a further seventy epochs to add 0.027. The early-stopping criterion never triggered, indicating that the run was capacity-limited rather than schedule-limited.

Final validation on the best checkpoint, over 7,399 images containing 172,375 instances, returned precision 0.670, recall 0.411, mAP@50 0.458 and mAP@50–95 0.289. The per-class breakdown is given in Table 3.4.

**Table 4 — YOLOv8n per-class validation results**

| **Class**     | **Images** | **Instances** | **Precision** | **Recall** | **mAP@50** | **mAP@50–95** |
|---------------|------------|---------------|---------------|------------|------------|---------------|
| Person        | 4,406      | 22,771        | 0.713         | 0.444      | 0.513      | 0.305         |
| Car           | 5,106      | 35,833        | 0.758         | 0.546      | 0.617      | 0.420         |
| Bicycle       | 1,209      | 2,632         | 0.662         | 0.337      | 0.401      | 0.225         |
| Motorcycle    | 1,062      | 2,342         | 0.648         | 0.460      | 0.504      | 0.294         |
| Bench         | 358        | 666           | 0.616         | 0.176      | 0.207      | 0.148         |
| Fire hydrant  | 705        | 788           | 0.779         | 0.454      | 0.516      | 0.379         |
| Traffic light | 2,469      | 16,669        | 0.684         | 0.333      | 0.389      | 0.198         |
| Stairs        | 174        | 219           | 0.595         | 0.543      | 0.581      | 0.374         |
| Pole          | 3,586      | 89,785        | 0.617         | 0.179      | 0.222      | 0.107         |
| Dog           | 505        | 670           | 0.625         | 0.636      | 0.630      | 0.436         |

Independent evaluation on the held-out test split of 3,699 images containing 86,381 instances produced precision 0.6692, recall 0.4086, mAP@50 0.4528, mAP@75 0.2954 and mAP@50–95 0.2847. The close agreement between validation and test figures, a gap of 0.005 in mAP@50, confirmed that the model wasn't overfitted to the validation split and that the evaluation protocol was sound. Inference speed was 0.8 ms per image at 640×640 on the A100, with 0.6 ms preprocessing and 1.0 ms post-processing.

### **Performance analysis**

The defining characteristic of this baseline is the divergence between precision and recall. At 0.670 precision and 0.411 recall, the detector was substantially more reliable when it spoke than it was attentive to what was present. For a navigation aid this asymmetry is precisely the wrong way round: a false positive causes an unnecessary pause, whereas a false negative may cause a collision.

The low recall wasn't uniform. Poles achieved recall of 0.179 despite constituting the single largest class in the dataset with 89,785 validation instances, a decisive demonstration that the problem is structural rather than a simple shortage of examples. Poles are thin, vertically elongated, low-texture, frequently occluded and often clustered, and at 640×640 with stride-8 as the finest prediction level they occupy very few feature cells. Benches at 0.176 recall and traffic lights at 0.333 exhibited the same small-or-thin-object pattern. Conversely, dogs achieved recall of 0.636 from only 670 instances, and stairs 0.543 from 219. Classes that are large, textured and visually distinctive succeed even when rare. Instance count and detection difficulty were, in other words, close to orthogonal.

### **Strengths and weaknesses**

The baseline's strengths were real. It converged stably without instability or large fluctuations, generalized consistently from validation to test, ran at approximately 0.8 ms per image, occupied only 6.3 MB as a stored checkpoint, and, most importantly, produced trustworthy per-class metrics, transforming the project from one that could observe losses into one that could diagnose behavior.

Its weaknesses were equally clear. Overall recall of 0.411 was insufficient for safety-critical use. Performance on thin and small structures was poor. And, decisively for the project's purpose, the ten-class vocabulary omitted most true pedestrian hazards: no bollards, no curbs, no potholes, no crossings, no bins, no construction barriers. A detector that reports cars and people accurately while remaining blind to the bollard in the user's path does not solve the navigation problem.

Improvement was therefore necessary along two axes simultaneously. Because the low recall was concentrated in classes that were structurally hard rather than merely rare, and because the missing vocabulary could only be supplied by new annotated data, the natural next step was to attack the dataset rather than the architecture.

## **3.3.5 Stage III – Dataset Expansion, Fine-Tuning and Optimization**

This stage constituted a single continuous optimization campaign rather than a sequence of independent trials. Its aim was to extend the detector's vocabulary and lift its performance on weak classes, and its trajectory was driven throughout by the diagnostic results of each preceding step. It produced the project's strongest per-class figures to date and, in the same campaign, its most serious failures.

### **Dataset expansion and class remapping**

Seven additional single-class datasets were collected as compressed archives and extracted into the working environment. Each had been annotated independently with its target object assigned class index 0, so all required remapping into the SeeSense label space. Their composition is given in Table 3.5.

**Table 5 — Auxiliary dataset composition (annotation counts)**

| **Source dataset** | **Train** | **Validation** | **Test** | **Assigned class ID** |
|--------------------|-----------|----------------|----------|-----------------------|
| Bicycle            | 1,582     | 446            | 230      | 2 (bicycle)           |
| Bollard            | 3,051     | 879            | 434      | 10 (bollard)          |
| Crosswalk          | 961       | 291            | 134      | 11 (crosswalk)        |
| Dogs               | 3,630     | 620            | 300      | 9 (dog)               |
| Pothole            | 3,277     | 377            | 399      | 12 (pothole)          |
| Scooter            | 1,411     | 62             | 51       | 13 (scooter)          |
| Stairs             | 1,687     | 200            | —        | 7 (stairs)            |

Three datasets reinforced existing classes (bicycle, dog, stairs) while four introduced new ones (bollard, crosswalk, pothole, scooter), expanding the vocabulary from ten to fourteen classes. A remapping routine copied each image under a source-prefixed filename and rewrote every annotation line's leading class index to the assigned target value, which both prevented filename collisions across sources and preserved provenance in the merged directory listing. The stairs dataset provided no test split, and the scooter dataset contributed only 62 validation and 51 test instances, an imbalance that would shortly become the limiting factor on that class. The resulting "final dataset" contained 8,176 training images with 8,176 matching label files, 1,598 validation images and 784 test images.

### **Fine-tuning on the expanded vocabulary**

Fine-tuning initialized from the Stage II best checkpoint. Ultralytics reconfigured the detection head from ten to fourteen outputs while retaining backbone and neck weights, producing a model of 3,008,378 parameters and 8.1 GFLOPs. Training ran for 30 epochs at 640×640 with batch size 16 and an explicitly reduced initial learning rate of 0.0001, an order of magnitude below the baseline's 0.01, chosen to adapt the representation without destroying it. The run completed in 0.421 hours on A100 across 511 iterations per epoch.

The results were, on their face, excellent. Validation over 1,598 images containing 2,875 instances returned precision 0.870, recall 0.804, mAP@50 0.850 and mAP@50–95 0.600. The per-class figures in Table 3.6 show several classes at levels far above anything achieved in Stage II.

**Table 6 — Validation results after fine-tuning on the expanded dataset**

| **Class** | **Images** | **Instances** | **Precision** | **Recall** | **mAP@50** | **mAP@50–95** |
|-----------|------------|---------------|---------------|------------|------------|---------------|
| Bicycle   | 394        | 446           | 0.969         | 0.980      | 0.988      | 0.770         |
| Stairs    | 183        | 200           | 0.898         | 0.900      | 0.918      | 0.671         |
| Dog       | 266        | 620           | 0.906         | 0.921      | 0.945      | 0.660         |
| Bollard   | 367        | 879           | 0.882         | 0.974      | 0.947      | 0.726         |
| Crosswalk | 220        | 291           | 0.919         | 0.869      | 0.934      | 0.705         |
| Pothole   | 124        | 377           | 0.767         | 0.515      | 0.647      | 0.328         |
| Scooter   | 26         | 62            | 0.749         | 0.468      | 0.571      | 0.338         |

Evaluation on the test split of 784 images containing 1,548 instances gave precision 0.862, recall 0.818, mAP@50 0.853 and mAP@50–95 0.618, in close agreement with validation.

![](media/image5.png)

![](media/image6.png)

![](media/image7.png)

### **Diagnosis: catastrophic forgetting**

These figures cannot be compared with the Stage II baseline and understanding why is one of the main findings of this stage. The instance-count array returned by the validator records zero instances for person, car, motorcycle, bench, fire hydrant, traffic light and pole. Only seven of the fourteen declared classes appear in the per-class table because the remaining seven do not occur anywhere in the expanded dataset. It was assembled exclusively from the auxiliary single-class sources and contains none of the original imagery.

The consequence is twofold. First, the headline mAP@50 of 0.850 is computed over seven comparatively easy classes on a dataset in which the average image contains fewer than two objects, whereas the Stage II figure of 0.458 was computed over ten classes on a dataset averaging more than twenty objects per image. The two numbers measure different things and their comparison is meaningless.

Second, and far more seriously, thirty epochs of gradient descent on data in which person, car, pole and traffic light never appear provides a continuous training signal that these classes are absent from every image. The classification head was actively driven to suppress them.

Diagnostic inference confirmed the concern. Running prediction over the expanded test set and tallying the resulting class assignments produced detections for bollard (519), crosswalk (133), pothole (329) and scooter (51) alone, with none of the original ten-class vocabulary appearing. The model had become an excellent seven-class detector at the cost of the ten-class capability that Stage II had spent 6.243 GPU-hours acquiring. This is catastrophic forgetting in its classical form, and it establishes the requirement that guided the rest of the project: vocabulary extension must be performed on data containing all classes simultaneously, never by sequential fine-tuning on separate class subsets.

### **Targeted oversampling of weak classes**

Within the expanded vocabulary, two classes remained clearly weak: pothole at 0.647 mAP@50 with recall 0.515, and scooter at 0.571 with recall 0.468. Both were the least-represented classes in the validation split, scooter especially so at 62 instances. Since acquiring additional annotated data was not available within the project's constraints, duplication-based oversampling was applied: every image whose label file contained an instance of class 12 or 13 was written to the output dataset three times in total under distinct filenames. This raised the training split from 8,176 to 11,384 images.

Fine-tuning proceeded from the previous checkpoint for 15 epochs at batch size 16 with a requested initial learning rate of 0.0001. This run executed on a Tesla T4 rather than the A100, completing in 0.875 hours, a useful practical illustration of hardware variability in shared cloud environments, with per-image inference rising from approximately 0.5 ms to 2.1 ms on the slower device. It is also worth recording that the automatic optimizer mode explicitly overrode the requested learning rate, selecting its own value. The intended learning rate reduction was therefore not applied as specified, a discrepancy visible only in the training log.

Test set results after oversampling, over 1,082 images containing 2,448 instances, gave precision 0.857, recall 0.843, mAP@50 0.866 and mAP@50–95 0.630. The targeted classes improved as intended: scooter rose from mAP@50 0.718 to 0.774 and mAP@50–95 from 0.507 to 0.568, with recall improving from 0.686 to 0.766, while pothole moved from 0.616 to 0.619 in mAP@50 with recall rising from 0.476 to 0.526.

The interpretation of these gains requires care, and the limitation is instructive. The duplication routine was applied uniformly across the training, validation and test splits, so the evaluation sets themselves were enlarged, validation from 1,598 to 1,898 images and test from 784 to 1,082 with duplicated copies of the very images containing the targeted classes. This alters the class weighting of the evaluation set and means the post-oversampling figures are not measured on the same benchmark as the pre-oversampling figures. The scooter improvement is supported by a genuine recall gain on a per-instance basis and is likely real, but the correct methodology is to oversample the training split alone and hold the evaluation splits fixed. This was recorded as a protocol defect to be corrected in the final stage.

The general lesson is that duplication-based oversampling has a limited potential. Repeating the same 62 scooter instances three times supplies no new visual variation. It only reweights the loss. It moves a badly under-represented class from unusable to marginal, but it cannot substitute for genuine data collection.

### **The failed combined-dataset experiment**

The forgetting diagnosis pointed to an clear solution: train on the union of the original ten-class dataset and the expanded set, so that all fourteen classes are present in every epoch. A merge routine was written to copy both source datasets into a unified directory under distinguishing filename prefixes.

The routine reported the following on execution: the three splits of the old dataset were copied successfully, and all three splits of the new dataset were skipped with the message that the folder was missing. The verification step that followed reported the merged dataset containing 25,892 training images, 7,399 validation and 3,699 test, figures identical to the original ten-class dataset. The oversampled dataset had been produced in an earlier session and no longer existed in the temporary working environment. The merge routine handled the missing directory by printing a notice and continuing.

The consequence was that a fifty-epoch training run was launched on a dataset believed to contain fourteen classes that in fact contained only the original ten. The configuration file declared "nc: 14", so the model was constructed with fourteen output classes, four of which had no instances anywhere in the data. Training proceeded from the previous checkpoint at an initial learning rate of 0.0002 and consumed 3.046 hours on the A100 without any error.

The results show the cost. Validation returned precision 0.669, recall 0.413, mAP@50 0.450 and mAP@50–95 0.279, with the instance-count array again recording zero instances for classes 10 through 13. Test-set evaluation gave precision 0.6709, recall 0.4041, mAP@50 0.4372 and mAP@50–95 0.2750. Table 3.7 sets these against the Stage II baseline on the identical test split.

**Table 7 — Failed combined run versus YOLOv8 baseline (identical test split)**

| **Metric** | **Stage II baseline** | **"Combined" run** | **Change** |
|------------|-----------------------|--------------------|------------|
| Precision  | 0.6692                | 0.6709             | + 0.0017  |
| Recall     | 0.4086                | 0.4041             | − 0.0045   |
| mAP@50     | 0.4528                | 0.4372             | − 0.0156   |
| mAP@50–95  | 0.2847                | 0.2750             | − 0.0097   |

Three GPU-hours produced a model measurably worse than the baseline it started from, having during the process also lost the four new classes acquired in the preceding experiments. The regression is consistent with a model that had been fine-tuned away from the ten-class distribution and then partially retrained back towards it, arriving at a compromise inferior to either endpoint.

### **Engineering lessons**

This stage produced four lessons that directly determined the design of the final stage.

The failure was silent, not loud. Every component behaved as designed: the copy routine reported the missing directory, the trainer accepted a configuration declaring more classes than the data contained, and the validator reported metrics for the classes that were present. No exception was raised at any point. The defect was not in any individual component but in the absence of a verification step between data preparation and training. A pipeline that allows a fifty-epoch run to begin on unverified data will eventually waste a fifty-epoch run.

Declared class count must be validated against observed class content. Constructing a fourteen-class head over a ten-class dataset is a detectable condition. It requires only counting the distinct class indices present in the label files and comparing against the configuration, and it should be a hard error rather than a silent acceptance.

Temporary storage is a correctness hazard, not merely an inconvenience. The oversampled dataset was lost between sessions, and its absence propagated into a corrupted experiment. Any artifact required by a subsequent step must be persisted, and its presence asserted before it is used.

Sequential fine-tuning on disjoint class subsets is not a viable vocabulary-extension strategy. The combination of the forgetting result and the merge failure established that the only reliable path to a seventeen-class detector was a single unified dataset in which every class appears, trained in one campaign.

## **3.3.6 Stage IV – YOLO26**

### **Motivation and continuity**

The final stage is the direct implementation of the requirements that Stage III produced. Each of its distinguishing features exists because a specific earlier failure demanded it: a single unified seventeen-class dataset because sequential fine-tuning caused catastrophic forgetting. A formal quality audit performed before training because the combined dataset run was destroyed by unverified data. Persistent per-epoch checkpointing with automatic resume because temporary storage had already cost the project a complete experiment. A fully documented, seed-fixed configuration because reproducibility had been informal, and per-class evaluation with latency benchmarking and deployment export because the model was now intended for the SeeSense system rather than for a notebook.

The move to YOLO26 as the architecture reflected the deployment profile. Its end-to-end detection head removes non-maximum suppression from the inference path, producing a fixed-size output tensor and eliminating a post-processing cost whose latency varies with scene density, a meaningful property for a real-time assistive device that must behave predictably in crowded environments as well as empty ones. The capacity increase from the nano-scale baseline to a small-scale model was justified by the vocabulary growing from ten to seventeen classes and the training set growing roughly threefold.

### **Dataset**

The unified dataset, was distributed as a single 11.7 GB archive to guarantee that every team member trained on an identical data version. It defines seventeen classes: person, car, bicycle, motorcycle, bench, fire hydrant, traffic light, stairs, pole, dog, curb, crosswalk, scooter, bollard, trash can, manhole and construction. Relative to the fourteen-class Stage III vocabulary, pothole was removed, while curb, trash can, manhole, and construction were added, resulting in the final seventeen-class vocabulary with the ground-level and street-furniture obstacles that the earlier iterations lacked.

The dataset comprises 74,096 training images, 11,086 validation images and 5,957 test images, totaling 91,139 images, with a matching label file for every image. It contains 693,146 training annotations, 180,752 validation annotations and 90,939 test annotations, for 964,837 objects overall. Compared with the 25,892 training images of Stage II, this represents a 2.9-fold increase in imagery and a substantially greater increase in annotation density.

Table 8 — Seventeen-class dataset composition

| Class         | Objects (all splits) | Images containing class |
|---------------|----------------------|-------------------------|
| Person        | 128,464              | 28,829                  |
| Car           | 192,524              | 32,307                  |
| Bicycle       | 15,647               | 8,071                   |
| Motorcycle    | 16,679               | 7,905                   |
| Bench         | 10,703               | 5,921                   |
| Fire hydrant  | 5,286                | 4,752                   |
| Traffic light | 82,892               | 12,733                  |
| Stairs        | 5,295                | 4,260                   |
| Pole          | 453,239              | 20,303                  |
| Dog           | 3,809                | 2,981                   |
| Curb          | 4,118                | 2,258                   |
| Crosswalk     | 4,223                | 3,511                   |
| Scooter       | 10,876               | 6,634                   |
| Bollard       | 6,597                | 2,699                   |
| Trash can     | 3,006                | 2,126                   |
| Manhole       | 722                  | 722                     |
| Construction  | 20,757               | 9,490                   |

![](media/image8.png)

The imbalance identified in Stage I persists and has in absolute terms intensified: poles now account for 453,239 objects, roughly 47% of all annotations, while manhole contributes 722 objects, a ratio exceeding 600: However, the distribution is materially healthier at the low end than in the ten-class dataset. Scooter, which had 62 validation instances in Stage III, now has 10,876 objects across 6,634 images. Bollard has grown from 879 validation instances to 6,597 objects. The classes that oversampling had attempted to support through artificial oversampling are now genuinely represented. The manhole class, appearing exactly once per image across 722 images, remained the clear weak point.

### **Dataset quality verification**

Before any training was permitted, the dataset was subjected to a systematic audit. The direct institutional response to the Stage III merge failure. Structural discovery confirmed the presence of all six required directories and verified that image and label counts matched exactly in each of the three splits.

Five categories of defect were then checked: empty annotation files, images lacking a corresponding label file, label files lacking a corresponding image, invalid annotations, and corrupted image files. The annotation validation examined every line of every label file across all 91,139 files, verifying that each contained exactly five fields, that the class index parsed to an integer within the valid range for the declared seventeen classes, that all four normalized coordinates parsed as floating-point values within [0, 1], and that no bounding box had zero or negative width or height. Image integrity was verified by opening and validating every image file.

The audit returned 0 defects in every category. This result is the substantive difference between Stage IV and everything preceding it: training began from a position of verified data integrity rather than assumed integrity. The audit was followed by visual inspection of twelve randomly sampled training images with their bounding boxes and class labels rendered, providing the qualitative check that automated validation cannot supply, correct class assignment, sensible box placement, and absence of systematic annotation drift.

![](media/image9.png)

![](media/image10.png)

![](media/image11.png)

### **Training environment and configuration**

Training was conducted in Google Colab on an NVIDIA A100-SXM4-40GB with 40,441 MiB of device memory, PyTorch 2.11.0 with CUDA 12.8, twelve CPU cores, 83.5 GB of system memory and Ultralytics 8.4.115. The session began with an explicit GPU availability assertion that raises an exception rather than silently falling back to CPU execution, a small but characteristic instance of the fail-loudly principle adopted after Stage III.

The configuration was defined explicitly and printed before execution rather than relying on framework defaults. Input resolution was 640×640, the schedule was 80 epochs with early-stopping patience of 20, batch size was 64, a fourfold increase over the baseline made possible by the A100's memory capacity, the optimizer was left in automatic mode and resolved to MuSGD at learning rate 0.01 with momentum 0.9 across 114 unregularised weight groups, 126 weight-decayed groups and 126 bias groups. Mixed-precision training was enabled and passed the automatic verification check, eight dataloader workers were used, and image caching was disabled given the dataset's size. Mosaic augmentation was closed over the final ten epochs. Reproducibility was enforced through a fixed seed of 42 and deterministic execution. Complete configuration was saved as JSON alongside the model artifacts.

The model was a YOLO26 small-scale model comprising 260 layers with 9,961,022 parameters and 22.8 GFLOPs during training, fusing at inference to 122 layers, 9,471,759 parameters and 20.8 GFLOPs. All 708 items transferred successfully from the pretrained checkpoint. The detection head was configured for seventeen classes over feature maps of 128, 256 and 512 channels with end-to-end operation enabled. One documentation mismatch should be recorded for accuracy: the run directory and the configuration variable were both named for the medium-scale variant, while the model actually trained, as confirmed by the architecture summary and parameter count in the training log, was the small-scale variant. The metrics reported below correspond to the small-scale model.

A notable architectural difference from YOLOv8 is visible in the training log itself: where the YOLOv8 run reported box, classification and distribution-focal loss terms, the YOLO26 run reports box, classification and L1 loss. This reflects the revised regression formulation of the newer architecture, and the L1 term operates on a numerically different scale, of order 10⁻³ rather than order 1, making cross-architecture loss comparison meaningless and reinforcing the necessity of comparing on metrics rather than losses.

### **Checkpoint management and training resumption**

The single most important technical change in this stage was persistent checkpointing. The output directory was placed on mounted Google Drive rather than in temporary session storage, and the save period was set to one, writing a complete checkpoint after every epoch. The weights directory accordingly contains eighty per-epoch checkpoints from 'epoch0.pt' through 'epoch79.pt', together with 'best.pt' and 'last.pt', each approximately 20.3 MB after optimizer stripping.

Training was governed by an explicit resume mechanism: the notebook tests for the existence of 'last.pt' in the run directory and, if found, reports the checkpoint, loads it and resumes, otherwise it initializes from pretrained weights and begins a new run. This structure makes the training cell idempotent across session boundaries, an essential property given that Colab sessions terminate on a schedule shorter than the total training time required.

The mechanism was used in practice. The executed session located the existing checkpoint and resumed training from epoch 61 to the eighty-epoch target, completing the remaining twenty epochs in 2.501 hours. The first sixty epochs had been completed in prior sessions. Without persistent checkpointing they would have been lost, exactly as the oversampled dataset had been lost in Stage III. Peak GPU memory during the resumed phase was approximately 19.6 GB.

### **Training process and results**

### The initial training session completed the first 60 epochs before the Google Colab session was interrupted. Training was then resumed from the saved checkpoint at epoch 61 and continued to the target of 80 epochs. The detailed epoch-by-epoch output from the initial session was not preserved, so Table 3.X reports the available metrics from epochs 61–80. The final twenty epochs are recorded in Table 3.9

**Table 9 — YOLO26 validation metrics, epochs 61–80**

| **Epoch** | **box_loss** | **cls_loss** | **l1_loss** | **Precision** | **Recall** | **mAP@50** | **mAP@50–95** |
|-----------|--------------|--------------|-------------|---------------|------------|------------|---------------|
| 61        | 1.700        | 1.255        | 0.00705     | 0.752         | 0.584      | 0.634      | 0.456         |
| 62        | 1.705        | 1.260        | 0.00703     | 0.752         | 0.587      | 0.635      | 0.457         |
| 63        | 1.704        | 1.256        | 0.00705     | 0.753         | 0.588      | 0.635      | 0.457         |
| 64        | 1.693        | 1.247        | 0.00700     | 0.749         | 0.589      | 0.636      | 0.458         |
| 65        | 1.689        | 1.251        | 0.00697     | 0.754         | 0.587      | 0.636      | 0.458         |
| 66        | 1.684        | 1.238        | 0.00690     | 0.754         | 0.587      | 0.636      | 0.458         |
| 67        | 1.677        | 1.227        | 0.00686     | 0.753         | 0.589      | 0.636      | 0.458         |
| 68        | 1.672        | 1.223        | 0.00681     | 0.757         | 0.586      | 0.635      | 0.458         |
| 69        | 1.671        | 1.218        | 0.00676     | 0.759         | 0.586      | 0.635      | 0.458         |
| 70        | 1.661        | 1.206        | 0.00672     | 0.752         | 0.589      | 0.635      | 0.457         |
| 71        | 1.711        | 1.187        | 0.00688     | 0.753         | 0.587      | 0.635      | 0.458         |
| 72        | 1.699        | 1.170        | 0.00673     | 0.756         | 0.587      | 0.634      | 0.458         |
| 73        | 1.688        | 1.158        | 0.00663     | 0.753         | 0.589      | 0.634      | 0.458         |
| 74        | 1.678        | 1.147        | 0.00654     | 0.756         | 0.587      | 0.633      | 0.458         |
| 75        | 1.673        | 1.138        | 0.00644     | 0.758         | 0.586      | 0.633      | 0.457         |
| 76        | 1.662        | 1.130        | 0.00636     | 0.758         | 0.586      | 0.633      | 0.457         |
| 77        | 1.652        | 1.117        | 0.00628     | 0.757         | 0.587      | 0.632      | 0.457         |
| 78        | 1.642        | 1.108        | 0.00621     | 0.759         | 0.585      | 0.631      | 0.456         |
| 79        | 1.639        | 1.101        | 0.00615     | 0.761         | 0.585      | 0.631      | 0.456         |
| 80        | 1.631        | 1.096        | 0.00609     | 0.763         | 0.584      | 0.631      | 0.456         |

Two behaviors in this training progress are worth noting. The first is convergence: mAP@50 varies only between 0.631 and 0.636 across the entire twenty-epoch window, a range of 0.005, while classification loss continues to fall steadily from 1.255 to 1.096. The model was, by epoch 61, extracting essentially all the performance available to it under this configuration, and the continuing loss reduction represents increasing confidence on examples already correctly handled rather than new capability. The eighty-epoch schedule was, in retrospect, longer than necessary.

The second is the mosaic-closure transition at epoch 71, visible as a sudden change in the box loss, which rises from 1.661 to 1.711 before resuming its decline. Disabling mosaic augmentation for the final ten epochs changes the training distribution to match the evaluation distribution more closely, and the losses reset onto a new scale. Notably, the closure did not improve validation mAP@50, which decreased slightly from 0.635 to 0.631 over the final ten epochs while precision rose from 0.753 to 0.763 and recall fell from 0.587 to 0.584, a small movement along the precision-recall trade-off rather than a genuine gain. The best checkpoint was accordingly selected from the mid-window epochs, not from the final epoch, which is the behavior best-checkpoint selection exists to provide. Early stopping at patience 20 was never triggered, and training concluded at the scheduled eighty epochs.

![](media/image12.png)

### **Validation results and per-class analysis**

Evaluation of the best checkpoint over the full validation split of 11,086 images containing 180,736 instances returned precision 0.7542, recall 0.5872, mAP@50 0.6356 and mAP@50–95 0.4581. The per-class breakdown is given in Table 3.10.

**Table 10 — YOLO26 per-class validation results**

| **Class**     | **Images** | **Instances** | **Precision** | **Recall** | **mAP@50** | **mAP@50–95** |
|---------------|------------|---------------|---------------|------------|------------|---------------|
| Person        | 5,083      | 24,474        | 0.706         | 0.495      | 0.553      | 0.340         |
| Car           | 5,615      | 37,057        | 0.744         | 0.590      | 0.644      | 0.449         |
| Bicycle       | 1,300      | 2,790         | 0.698         | 0.398      | 0.456      | 0.268         |
| Motorcycle    | 1,129      | 2,459         | 0.726         | 0.488      | 0.555      | 0.342         |
| Bench         | 665        | 1,201         | 0.749         | 0.402      | 0.463      | 0.328         |
| Fire hydrant  | 924        | 1,018         | 0.819         | 0.597      | 0.655      | 0.489         |
| Traffic light | 2,547      | 16,924        | 0.685         | 0.414      | 0.445      | 0.237         |
| Stairs        | 555        | 677           | 0.795         | 0.728      | 0.769      | 0.591         |
| Pole          | 3,806      | 90,122        | 0.670         | 0.205      | 0.256      | 0.128         |
| Dog           | 615        | 809           | 0.739         | 0.700      | 0.733      | 0.550         |
| Curb          | 76         | 238           | 0.588         | 0.206      | 0.289      | 0.156         |
| Crosswalk     | 169        | 190           | 0.900         | 0.937      | 0.957      | 0.774         |
| Scooter       | 400        | 584           | 0.904         | 0.913      | 0.950      | 0.853         |
| Bollard       | 198        | 769           | 0.812         | 0.821      | 0.864      | 0.631         |
| Trash can     | 267        | 397           | 0.785         | 0.754      | 0.832      | 0.684         |
| Construction  | 459        | 1,027         | 0.747         | 0.748      | 0.748      | 0.509         |

The manhole class does not appear in this table because the validation split contains no manhole instances. The per-class summary printed by the notebook reports a value of 0.4581 for manhole, but this is identical to the overall mAP@50–95 and is the framework's placeholder for a class with no validation representation rather than a measured score. Manhole performance is therefore unmeasured, which, given only 722 objects in the entire dataset, is an accurate reflection of that class's status.

The per-class results are divided into three clear groups. The strongest group comprises the newly introduced navigation-specific classes: scooter at mAP@50 0.950 with mAP@50–95 0.853, crosswalk at 0.957 and 0.774, bollard at 0.864 and 0.631, and trash can at 0.832 and 0.684. These are precisely the classes the project set out to add, and they are detected with a much higher reliability than the general urban classes. Their high mAP@50–95 values indicate that localization remains strong even at stricter IoU thresholds. The comparison with Stage III is instructive: scooter, which oversampling had lifted only to 0.774 mAP@50 with 0.568 mAP@50–95 on a duplicate-inflated test set, now reaches 0.950 and 0.853 on a clean split with genuine data, a direct demonstration that real examples accomplish what duplication cannot.

The middle group contains the classic urban objects: fire hydrant at 0.655, car at 0.644, motorcycle at 0.555, person at 0.553. These improved over Stage II under their respective validation protocols, and the improvement is broadly consistent across the group. Person remains limited by recall of 0.495, reflecting the high frequency of heavily occluded and distant pedestrians in urban imagery.

The weakest group is unchanged in composition from Stage II. Pole reaches mAP@50 0.256 with recall 0.205, curb 0.289 with recall 0.206, and traffic light 0.445 with recall 0.414. Poles constitute 90,122 of the 180,736 validation instances, half the entire validation object population, yet remain the hardest class in the vocabulary. Curbs, at 238 validation instances, suffer from both scarcity and intrinsic difficulty: a kerb is a low-contrast linear boundary whose extent is difficult to define clearly, and axis-aligned bounding boxes are a poor representation for such geometry. These two classes are the main limitation on overall recall, and their persistence across every architecture tried in this project establishes them as a property of the object geometry rather than of any model.

![](media/image13.png)

![](media/image14.jpeg)

![](media/image15.jpeg)

### **Inference performance**

Latency was measured under two conditions. Validation over the full split at batch scale recorded 0.6 ms preprocessing, 1.4 ms inference and 0.1 ms post-processing per image. A field-test pass over 100 randomly selected test images at a confidence threshold of 0.25 recorded 1.8 ms preprocessing, 1.1 ms inference and 0.2 ms post-processing, for approximately 3.1 ms per image end to end, corresponding to roughly 900 frames per second for the inference stage alone on the A100.

The post-processing figure of 0.1–0.2 ms is important, because it is the practical benefit of the end-to-end detection head. The YOLOv8 baseline required 1.0–1.2 ms of post-processing per image, comparable to or exceeding its own inference time, because non-maximum suppression cost scales with the number of candidate boxes and therefore with scene density. The YOLO26 model produced a fixed-size output tensor of shape (1, 300, 6) with no suppression stage, so its post-processing cost is both an order of magnitude lower and, more importantly for a real-time assistive device, effectively constant regardless of how crowded the scene is. A detector whose latency is stable between an empty pavement and a busy intersection is materially more suitable for a safety-critical application than one whose worst case coincides with its most demanding scenario.

The field test also provided qualitative confirmation of behavior across vocabulary. Detections spanned all seventeen classes with reasonable detection patterns, dense multi-object street scenes returning combinations such as thirteen cars, seventeen traffic lights and seven poles in a single frame, alongside isolated single-object images returning one construction marker or one trash can. Two of the hundred images returned no detections at the 0.25 threshold.

### **Export**

The best checkpoint was exported to ONNX (Open Neural Network Exchange) using Opset 20, with ONNXSlim optimization applied. Opset defines the version of ONNX operations and features available during model export, while ONNXSlim simplifies and optimizes the exported computational graph without changing the model's behavior. The export took 5.3 seconds and produced a 36.4 MB ONNX artifact from the 19.4 MB PyTorch checkpoint. The exported graph accepts input of shape (1, 3, 640, 640) and produces output of shape (1, 300, 6), three hundred candidate detections each described by four box coordinates, a confidence score and a class index, with no post-processing required. This fixed-shape, framework-independent artifact is directly deployable to ONNX Runtime, TensorRT and mobile inference engines, completing the transition from research notebook to deployable component. The complete artifact set was persisted to Drive: the ONNX export, the best and last checkpoints, all eighty per-epoch checkpoints, the serialized training configuration, a metrics summary in JSON, and the full complement of generated diagnostic plots including confusion matrices in raw and normalized form, precision, recall, F1 and precision-recall curves, and paired label-versus-prediction batch visualizations.

### **Engineering challenges and performance discussion**

Three engineering challenges dominated this stage. The first was session instability. Colab sessions terminate well before eighty epochs of training on 74,096 images can complete, and the resume mechanism was not a convenience but necessary for completing the training. That the executed session resumed from epoch 61 and completed the remaining twenty epochs in 2.501 hours implies a total training cost of roughly ten GPU-hours spread across multiple sessions, an amount that could not have been accumulated without per-epoch persistence to durable storage.

The second was the storage cost of that persistence. Eighty checkpoints at 20.3 MB each consume approximately 1.6 GB of Drive capacity for a single run. The saving period of one was a deliberate trade of storage for protection against data loss, justified by the demonstrated cost of losing work. A production configuration might reasonably save less frequently once the pipeline's stability is established.

The third was the persistence of the imbalance problem across every scale of intervention attempted. Poles were the dominant class in Stage I and caused class collapse. They were the worst performing class in Stage II. They remain the worst-performing class in Stage IV at recall 0.205, despite a 44% increase in absolute pole annotations and substantially more capable architecture. This is now understood not as a data-quantity problem but as a property of the object class: thin vertical structures with minimal texture, frequently occluded and clustered, are naturally difficult for axis-aligned bounding-box detection at stride-8 resolution.

Set against these constraints, the overall performance improvement is substantial. Precision rose from 0.670 to 0.754 and recall from 0.411 to 0.587, the latter a 43% relative improvement in the metric that matters most for a safety application, achieved while simultaneously expanding the vocabulary from ten to seventeen classes. Both directions of movement are ordinarily in tension: adding seven classes typically depresses per-class performance through increased inter-class confusion, and improving recall typically costs precision. Achieving both simultaneously indicates that the combination of a substantially larger verified dataset and a more capable architecture produced a genuine capability gain rather than a repositioning along an existing trade-off curve.

### **Final analysis**

The Stage IV model satisfies the requirements established at the outset of the project. Its vocabulary covers the navigation-hazard categories that motivated building a dedicated detector, and the classes added for that purpose, crosswalk, scooter, bollard, trash can are among its strongest, all exceeding mAP@50 of 0.83. Its inference cost of approximately 1.1–1.4 ms per image with near-constant post-processing supports real-time operation. Its training was reproducible, its data verified, its artifacts saved, and its output exported to a deployment-ready format.

Its limitations are equally well understood, which is itself a product of the evaluation discipline established after Stage I. Overall recall of 0.587 means the detector still misses a substantial proportion of annotated objects, and the low recall is concentrated in poles and curbs. The manhole class has too few examples to be meaningfully evaluated. These are known, measured constraints rather than unknown risks, and they are documented as such.

![](media/image16.jpg)

![](media/image17.jpg)

![](media/image18.jpg)

![](media/image19.jpg)

## **3.3.7 Comparative Analysis**

### **Overall performance**

Table 11 consolidates the principal results of the four stages. Comparison across stages requires care, because the datasets and class vocabularies differ, and the table records those differences explicitly rather than eliding them.

**Table 11 — Consolidated comparison across all four stages**

|                      | **Stage I: Custom**             | **Stage II: YOLOv8n** | **Stage III: fine-tuned** | **Stage III: combined (failed)** | **Stage IV: YOLO26**      |
|----------------------|---------------------------------|-----------------------|---------------------------|----------------------------------|---------------------------|
| Architecture         | ResNet-18 + custom head         | YOLOv8n               | YOLOv8n                   | YOLOv8n                          | YOLO26s                   |
| Parameters           | ~11.2M backbone (frozen) + head | 3,012,798             | 3,008,378                 | 3,008,378                        | 9,961,022                 |
| Classes              | 10                              | 10                    | 14 declared / 7 present   | 14 declared / 10 present         | 17                        |
| Training images      | 25,892                          | 25,892                | 8,176 → 11,384            | 25,892                           | 74,096                    |
| Epochs completed     | 7 (early stop)                  | 100                   | 30 + 15                   | 50                               | 80                        |
| Training time        | —                               | 6.243 h               | 0.421 h + 0.875 h         | 3.046 h                          | 2.501 h (final 20 epochs) |
| Precision            | not measured                    | 0.670                 | 0.870 / 0.846             | 0.669                            | 0.754                     |
| Recall               | not measured                    | 0.411                 | 0.804 / 0.819             | 0.413                            | 0.587                     |
| mAP@50               | not measured                    | 0.458                 | 0.850 / 0.859             | 0.450                            | 0.636                     |
| mAP@50–95            | not measured                    | 0.289                 | 0.600 / 0.610             | 0.279                            | 0.458                     |
| Inference (ms)       | —                               | 0.8                   | 0.5 (A100) / 2.1 (T4)     | 0.5                              | 1.4                       |
| Post-processing (ms) | manual NMS                      | 1.0–1.2               | 1.0                       | 1.2                              | 0.1–0.2                   |

The Stage III figures are shown for both the initial fine-tuning and the post-oversampling run, and they must be read with the qualification established in Section 3.5: they are computed over seven classes on a sparse dataset, and the post-oversampling evaluation splits contain duplicated images. They do not represent a detector superior to Stage II or Stage IV, they represent a narrow detector measured on a narrow benchmark.

### **Precision, recall and mean average precision**

The most informative comparison is between Stage II and Stage IV, since both are full-vocabulary models evaluated over their complete class sets under equivalent protocols. Precision improved from 0.670 to 0.754, a relative gain of 12.5%. Recall improved from 0.411 to 0.587, a relative gain of 42.8%. mAP@50 improved from 0.458 to 0.636 and mAP@50–95 from 0.289 to 0.458, the latter a relative gain of 58.5%.

The improvement in mAP@50–95 was larger than the improvement in mAP@50. Because mAP@50–95 evaluates performance across multiple IoU thresholds up to 0.95, it is more sensitive to bounding-box localization accuracy. This suggests that the Stage IV model improved not only in detecting objects, but also in placing more accurate bounding boxes around them. This improvement is consistent with the updated regression method and the larger, verified training set.

Table 12 compares the ten classes common to both models under their respective validation protocols. The validation sets differ in size and composition, so these figures indicate direction and magnitude of change rather than a controlled ablation.

**Table 12 — Shared classes: Stage II versus Stage IV validation mAP**

| **Class**     | **Stage II mAP@50** | **Stage IV mAP@50** | **Stage II mAP@50–95** | **Stage IV mAP@50–95** |
|---------------|---------------------|---------------------|------------------------|------------------------|
| Person        | 0.513               | 0.553               | 0.305                  | 0.340                  |
| Car           | 0.617               | 0.644               | 0.420                  | 0.449                  |
| Bicycle       | 0.401               | 0.456               | 0.225                  | 0.268                  |
| Motorcycle    | 0.504               | 0.555               | 0.294                  | 0.342                  |
| Bench         | 0.207               | 0.463               | 0.148                  | 0.328                  |
| Fire hydrant  | 0.516               | 0.655               | 0.379                  | 0.489                  |
| Traffic light | 0.389               | 0.445               | 0.198                  | 0.237                  |
| Stairs        | 0.581               | 0.769               | 0.374                  | 0.591                  |
| Pole          | 0.222               | 0.256               | 0.107                  | 0.128                  |
| Dog           | 0.630               | 0.733               | 0.436                  | 0.550                  |

Every shared class improved. The largest gains occurred in bench, which more than doubled from 0.207 to 0.463, and stairs, which rose from 0.581 to 0.769. These were also classes whose training representation grew substantially in the unified dataset, from 2,152 to 10,703 annotations for bench and from 836 to 5,295 for stairs. The smallest gain occurred in pole, which improved by only 0.034 despite already being the most abundant class, confirming that its difficulty is structural. The pattern across the table is consistent: classes limited by data availability improved markedly, while classes limited by object geometry improved marginally.

### **Engineering complexity**

Engineering complexity did not increase consistently across the stages, it peaked at Stage I and fell thereafter. The custom detector required manual implementation of the target encoder, the composite loss, the decoding routine, the intersection-over-union computation, non-maximum suppression, and all visualization, with no evaluation harness produced despite that effort. Stage II replaced all of it with framework calls, and the engineering effort shifted from implementing detection mechanics to preparing data and interpreting results, which is where it delivered value.

Complexity then reappeared at a higher level of abstraction. Stage III required dataset acquisition, class remapping across seven heterogeneous sources, merge logic and oversampling routines, and it was precisely in this custom data-handling code that the project's most expensive failure occurred. Stage IV re-invested that complexity into verification and infrastructure: a five-category quality audit, explicit configuration management, checkpoint persistence and resume logic. The trajectory is instructive. Complexity spent on re-implementing solved algorithmic problems produced an unevaluable model. Complexity spent on verifying data and securing infrastructure produced a deployable one.

### **Training stability**

Stability improved consistently. Stage I terminated by early stopping at epoch seven with validation loss plateaued and training loss still falling, the classic overfitting signature, complicated by the fact that the model was simultaneously collapsing to a single class. Stage II converged smoothly over 100 epochs after the expected transient warm-up, with consistent improvement and no large fluctuations, though it exhibited severely diminishing returns beyond epoch 30. Stage III's fine-tuning runs were stable and rapid, converging within 30 and 15 epochs respectively likely because they started from pretrained weights and were trained on a narrower dataset. Stage IV was the most stable of all: validation mAP@50 varied by only 0.005 across the final twenty epochs, and the single visible change, the mosaic-closure discontinuity at epoch 71, was a deliberate scheduled intervention rather than an instability.

### **Dataset influence**

The single most consistent finding across all four stages is that dataset composition determined outcomes more strongly than architecture. Stage I failed partly because the target encoding could not represent the object density of the data. Stage II's per-class results correlated with class characteristics rather than with class frequency. Stage III demonstrated the point twice over, first by showing that fine-tuning on data lacking seven classes destroys those seven classes, and second by showing that a fifty-epoch run on unverified data produces a regression regardless of how correct the training configuration is. Stage IV's improvements are attributable in substantial part to a 2.9-fold larger, audited, unified dataset in which previously scarce classes are genuinely represented.

The contrast between the scooter results makes the argument briefly. In Stage III, scooter had 1,411 training instances and 62 validation instances; duplication-based oversampling raised its test mAP@50 to 0.774 on an inflated benchmark. In Stage IV, with 10,876 genuine scooter objects across 6,634 images, the class reaches 0.950 mAP@50 and 0.853 mAP@50–95 on a clean split. Real data achieved what duplication could only approximate.

### **Generalization**

Generalization behavior also improved across the stages and was measured rather than assumed from Stage II onward. Stage I offered no generalization evidence beyond a validation loss plateau. Stage II demonstrated genuine generalization with a validation-to-test gap of only 0.005 in mAP@50 (0.458 against 0.4528) and 0.004 in mAP@50–95, indicating that the model had learned transferable features rather than memorized the validation split. Stage III's fine-tuned models showed similarly close validation-test agreement, 0.850 against 0.853, and 0.859 against 0.866, but within a narrow domain that does not test generalization meaningfully. Stage IV was validated on 11,086 images containing 180,736 instances, a validation population larger than the entire training set of Stage III, providing correspondingly stronger evidence.

### **Advantages and disadvantages**

Each stage carried a distinct profile.

Stage I offered complete transparency and educational value at the cost of poor performance, no metrics, and unrecoverable architectural limits.

Stage II offered stability, speed, a small footprint and trustworthy metrics, at the cost of low recall and limited class vocabulary.

Stage III offered strong per-class performance on the new classes and demonstrated that vocabulary extension was achievable, at the cost of catastrophic forgetting, evaluation-split contamination through oversampling applied to all dataset splits, and a silently corrupted merge.

Stage IV offered the full vocabulary, the best full-vocabulary metrics, verified data, near-constant post-processing latency and a deployment-ready export, at the cost of a threefold larger model, substantial training time and storage, and unresolved weakness on poles, curbs and manholes.

### **Stage-to-stage improvement**

Each transition was driven by a specific measured deficiency.

Stage I → Stage II replaced an architecture that could not represent dense scenes, could not localize small objects and could not be evaluated, with one that solved all three structurally, converting an unmeasurable model into a baseline of mAP@50 0.458.

Stage II → Stage III responded to insufficient recall and limited class vocabulary by attacking the data, extending the vocabulary from ten to fourteen classes and reaching mAP@50 above 0.85 on the new classes while revealing, through catastrophic forgetting and a corrupted merge, exactly which methods do not work.

Stage III → Stage IV consolidated those lessons into a unified seventeen-class audited dataset trained in a single campaign with verified integrity and persistent checkpointing, producing simultaneous improvement in precision, recall and both mAP measures over the last full-vocabulary baseline, together with a substantial reduction in post-processing latency and a deployable artifact.

## **3.3.8 Final Conclusions**

This chapter has documented the development of the SeeSense object detector through four stages, from a transparent custom prototype to a validated, exported seventeen-class final model. The progress was not a planned progression through architectures. It was a sequence of responses to specific, measured failures.

The detector evolved along three simultaneous dimensions. Architecturally, it moved from a frozen ResNet-18 with a single-scale hand-written head, through a multi-scale anchor-free YOLOv8n at 3.0 million parameters, to a YOLO26 small-scale model at 9.5 million fused parameters with an end-to-end detection head. In terms of vocabulary, it grew from ten general urban classes to seventeen classes covering the navigation hazards that define the application. In terms of data, it grew from 25,892 unaudited training images to 74,096 verified images carrying 693,146 annotations. The final model achieves precision 0.754, recall 0.587, mAP@50 0.636 and mAP@50–95 0.458 across all seventeen classes, with approximately 1.1–1.4 ms inference and 0.1–0.2 ms post-processing per image, exported to ONNX for deployment.

Each stage provided findings that directly influenced the design of the next stage, and this is not a retrospective justification: Stage I established, through direct measurement rather than assumption, the specific architectural properties the task demands: multi-scale prediction, dynamic label assignment, IoU-aligned regression, and above all a proper evaluation metric. Its most valuable output was not a model but the recognition that a training pipeline reporting only loss cannot detect its own failure. The model had collapsed to predict a single class while its loss curves suggested success. Stage II converted the project from one that observed losses into one that could diagnose behavior per class, and its diagnosis, that recall was the binding constraint and that the low recall lay in small and thin structures, directed everything that followed. Stage III proved the vocabulary could be extended and simultaneously proved that the obvious method of extending it does not work. The catastrophic forgetting result and the silent merge failure are the reason Stage IV was designed as a single unified training campaign preceded by a mandatory quality audit. Stage IV was the implementation of everything the first three stages had established.

The final model was selected based on measured criteria. It is the only model in the sequence that covers the complete seventeen-class navigation vocabulary and achieves the best full-vocabulary metrics. Compared with the last comparable baseline, recall improved by 42.8% and mAP@50–95 by 58.5%. The navigation-specific classes for which the detector was developed are among its strongest, with crosswalk, scooter, bollard, and trash can all exceeding an mAP@50 of 0.83. The model also provides significantly lower post-processing latency and was trained on verified data under a fully documented configuration, with saved artifacts and a deployment-ready export.

Five engineering lessons emerged with sufficient force to be considered the chapter's durable contribution. The first is that a training pipeline must be able to detect its own failure: metrics that measure the objective being optimized are not a substitute for metrics that measure the capability being sought, and the Stage I collapse was invisible precisely because only the former were recorded. The second is that data verification must precede training and must be a hard gate: the combined dataset run consumed three GPU-hours and produced a regression because a missing directory was reported rather than raised, and a declared class count was accepted without being checked against the classes present. The third is that class imbalance is a first-order design constraint that scales with the dataset rather than being solved by it, poles were the dominant class and the worst-performing class in every stage of this project, and increasing their absolute count by 44% moved their mAP@50 by 0.034. The fourth is that vocabulary extension by sequential fine-tuning on disjoint class subsets causes catastrophic forgetting, and that the only reliable method is a unified dataset in which all classes are present in every epoch. The fifth is that infrastructure is not separable from research: temporary storage corrupted an experiment, and the eighty-epoch training run that produced the final model existed only because per-epoch checkpoints were written to durable storage and an automatic resume path was implemented.

The research delivered a detector that meets the main functional requirements of the project while also providing a clear understanding of its remaining limitations. More importantly, the development process established a reliable approach for training, evaluating, and improving the model based on measured results rather than assumptions. The final model represents a significant improvement over the earlier stages in detection performance, class coverage, training reliability, and deployment readiness. Overall, the project demonstrates how continuous evaluation and lessons learned from each stage can guide the development of a more accurate, reliable, and practical object detection system for SeeSense.
