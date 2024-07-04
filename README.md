# ML_Project3: Breast Cancer Detection

## Project Description
ML_Project3 is a machine learning project focused on detecting breast cancer using datasets from sklearn. This project involves data preprocessing, model training, and deploying a machine learning model. The goal is to create a robust pipeline for data ingestion, transformation, model training, and prediction. Additionally, a web application is developed to facilitate user interaction and visualize predictions.

## Folder Structure

```
ML_Project3/
├── artifacts/
│   ├── test.csv
│   ├── model.pkl
│   ├── raw.csv
│   ├── train.csv
│   ├── preprocessor.pkl
├── .DS_Store
├── LICENSE
├── requirements.txt
├── gen.py
├── README.md
├── setup.py
├── .gitignore
├── app.py
├── templates/
│   ├── form.html
│   ├── index.html
│   ├── result.html
├── upload_data_db/
│   ├── upload.ipynb
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_train.ipynb
├── src/
│   ├── exception.py
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│   ├── logger.py
│   ├── components/
│   │   ├── model_trainer.py
│   │   ├── data_ingestion.py
│   │   ├── __init__.py
│   │   ├── data_transformation.py
│   ├── utils.py
```

## Files and Directories

- `artifacts/`: Contains intermediate files and final outputs like the trained model and preprocessor.
  - `test.csv`: Test dataset.
  - `model.pkl`: Trained model file.
  - `raw.csv`: Raw dataset.
  - `train.csv`: Training dataset.
  - `preprocessor.pkl`: Preprocessing object file.

- `LICENSE`: Project license file.

- `requirements.txt`: Contains the list of dependencies required for the project.

- `gen.py`: Script for data generation or processing.

- `README.md`: Project documentation file.

- `setup.py`: Script for setting up the project.

- `.gitignore`: Specifies files and directories to be ignored by Git.

- `app.py`: Main application script for running the web application.

- `templates/`: Contains HTML templates for the web application.
  - `form.html`: Form template for user input.
  - `index.html`: Main index page template.
  - `result.html`: Result display template.

- `upload_data_db/`: Contains notebooks related to data upload processes.
  - `upload.ipynb`: Notebook for uploading data to a database.

- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis (EDA) and model training.
  - `EDA.ipynb`: Notebook for exploratory data analysis.
  - `model_train.ipynb`: Notebook for model training.

- `src/`: Contains the source code for the project.
  - `exception.py`: Custom exception handling module.
  - `__init__.py`: Initialization file for the `src` module.
  - `pipelines/`: Contains the pipeline scripts.
    - `training_pipeline.py`: Script for the training pipeline.
    - `__init__.py`: Initialization file for the `pipelines` module.
    - `prediction_pipeline.py`: Script for the prediction pipeline.
  - `logger.py`: Logger module for logging.
  - `components/`: Contains various components of the project.
    - `model_trainer.py`: Module for model training.
    - `data_ingestion.py`: Module for data ingestion.
    - `__init__.py`: Initialization file for the `components` module.
    - `data_transformation.py`: Module for data transformation.
  - `utils.py`: Utility functions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML_Project3.git
   cd ML_Project3
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage

- **Data Ingestion**: The `data_ingestion.py` module is used to ingest data from various sources.
- **Data Transformation**: The `data_transformation.py` module is used to preprocess the data.
- **Model Training**: The `model_train.ipynb` notebook and `model_trainer.py` module are used for training the model.
- **Prediction**: The `prediction_pipeline.py` is used for making predictions with the trained model.
- **Web Application**: The web application allows users to input data and get predictions. It can be started by running `app.py`.
  
## Deployment

The project has been deployed and can be accessed at [https://ml-project3.onrender.com](https://ml-project3.onrender.com).

## Docker

 **Build and Run**
- To build and run the Docker container:
   ```bash
   docker build -t ml_project3 
   ```


- Build the Docker image:
  ```bash
   docker run -p 5000:5000 ml_project3
   ```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Special thanks to all contributors and open-source libraries used in this project.
  
## Contact

For any questions or issues, please contact [nishchalgaur2003@gmail.com](mailto:nishchalgaur2003@gmail.com).


---

Feel free to reach out if you have any questions or issues!
