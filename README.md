# Construction_progress_Automation
PROBLEM STATEMENT :
                Utilization of images for monitoring of progress of construction activities for building construction projects.
DATASET SOURCE:
                I  Collected dataset manually real data by capturing pictures by camera. Also collected from google images.
DATASET DESCRIPTION:
              The dataset consists of real-world images collected manually through camera captures and additional images sourced from Google. These images are organized into four primary categories related to the stages of building construction activities:
1.	Facade: Images depicting the external face of the building, including walls, windows, and architectural finishes.
2.	Superstructure: Images showing the building's load-bearing components such as beams, columns, and the main framework.
3.	Interior: Images representing interior spaces, including walls, floors, ceilings, and internal finishes during the construction process.
4.	Foundation: Images focused on the base of the building, including foundations, footings, and other substructure elements.
This dataset is designed to assist in the monitoring and analysis of construction progress by capturing various stages and components of building construction activities.

MODEL TRAINING :
       
The CNN model was trained using a dataset split into training and testing sets. The architecture included several convolutional layers followed by max-pooling and dense layers. The Adam optimizer was used with a learning rate of 0.001, and categorical cross-entropy served as the loss function. Data augmentation techniques, such as adding Gaussian noise and modifying brightness and contrast, were applied to prevent overfitting and improve generalization. The model was trained for 20 epochs with a batch size of 32, balancing learning performance and computational efficiency.

REGULARIZATION TECHNIQUES:
 Data Augmentation: 
              By applying transformations such as adding noise, adjusting brightness, and altering contrast to the input images, data augmentation served as a powerful regularization technique. It artificially increased the diversity of the training dataset without altering the labels, allowing the model to generalize better and become more robust to variations in input data. This process helps in preventing overfitting by reducing the model's dependence on specific features of the training images, thus improving its performance on unseen data.
RESULT:
After training the CNN model, the following results were obtained on the test dataset:
•	Test Accuracy: 78.65%
•	Test Loss: 0.4459
These results indicate that the model performed reasonably well, correctly classifying approximately 79% of the test images. The loss value of 0.4459 suggests that the model's predictions were fairly accurate.

