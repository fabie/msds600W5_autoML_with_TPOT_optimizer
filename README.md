# msds600W5_autoML_with_TPOT_optimizer
This is an example where TPOT can be used to optimize accurate predictions. TPOT is a supervised autoML. 
A simple example of autoML with TPOT Pipeline optimizer in Python.
Tree-based Pipeline Optimization Tool (TPOT)   is a Tree-Based Pipeline Optimization Tool for Automating Machine Learning.
TPOT is an open source programming-based AutoML system that optimizes a series of feature and machine learning models with the goal of maximizing classification accuracy on a supervised classification task. 
This is very useful for optimizing accurate predictions and the process is automatized.  Finding a model for optimizing prediction with accuracy is tedious and is time-intensive. Integrating Pareto principle 80:20 split between training and testing, TPOT can automatically generate and compact pipelines that consistently outperform basic machine learning. 

To use this repository:
(1) download the CSV file “prepped_churn_data”.
(2) download the python file “tpot_pchurn_data_pipeline_filledin.py”

Output the python file to a text file and display the top of the file: python tpot_pchurn_data_pipeline_filledin.py < text_filename.txt | head 

Or simply use the your favorite code editor (Sublime, Xcode, IDE)

Change the path to where you have located the files: tpot_data = pd.read_csv('/Users/.../.../.../.../')

fabie vc maintains this repository. The code was created on Anaconda and Python 3.9.13.
Python and conda libraries include: numpy, pandas, sklearn, tpot
