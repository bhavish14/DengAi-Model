
# Utilities
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ML Modules
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

# Metrics Modules
from sklearn.metrics import classification_report

# Plotting Modules
import seaborn as sns
import matplotlib.pyplot as plt


class dengAi():
	"""
    An encapsulation of the methods and other utility variables for the DengAi model.
    """
	def __init__(self, train_path, train_results_path, path_test):
		"""
        Construct a new 'DengAi' object.

        :param train_path: Absolute path of the training data
        :param train_results_path: Absolute path of training results
		:param path_test: Absolute path of the data on which the model is to be tested. 
        :return: returns nothing
        """

		# Training Data
		self.X_train = pd.read_csv(
			train_path,
		)
		self.y_train = pd.read_csv(
			train_results_path,
		)

		# Test Data
		self.X_test = pd.read_csv(
			path_test,
		)

		# Utility Variables
		self.correlation = []
		self.dataset = []
		self.test_dataset = []
		self.final_columns_list = []

		self.x_vals = []
		self.y_vals = []
		self.x_train_vals = []
		self.y_train_vals = []

		# Initialization for various models prediction result varaiables
		self.init_models()

	def init_models(self):
		"""
        Initializes the DengAi object fields with models for various Machine Learning 
		algorithms with pre-determined parameters.
        """
		self.scaler = StandardScaler()
		self.d_tree = DecisionTreeRegressor(
			criterion="mae", splitter="best", random_state=40
		)
		self.svc = SVC(
			decision_function_shape="ovr", probability=True
		)
		self.rand_forest = RandomForestRegressor(
			n_estimators= 20,
			criterion='mse',
			min_samples_split=2
		)
		self.adaboost = AdaBoostRegressor(
			n_estimators = 100,
			loss = 'exponential'
		)

		# Prediction Report for various models and parameter combinations
		self.report = {}

		# Prediction values of the best model
		self.train_predictions = {}
		self.test_predictions = {}

	# Data Preprocess

	def preprocess_data(self, test):
		"""
        Given a Dataset, preprocess_data subjects it to;
			- Null Value removal,
			- Converting categorical data to numerical values, 
			- Standardization, and
			- Converting all columns into a single data type.

        :param test: Data frame that is to be preprocessed.
		:return: None
		"""
		# Appending labels to the train dataset
		if test == False:
			X_train = pd.concat([self.X_train, self.y_train['total_cases']], axis=1)
		else:
			X_train = self.X_test

		random_DF = X_train.reindex(np.random.permutation(X_train.index))

		# Replacing null values with the column mean
		random_DF.fillna(method='ffill', inplace=True)

		# Converting categorical data to numerical
		encoder = LabelEncoder()
		random_DF["city"] = encoder.fit_transform(random_DF["city"])

		# Converting week start date into a season {months are mapped into seasons}
		seasons = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]
		seasons_series = []

		for item in random_DF['week_start_date']:
			seasons_series.append(
				seasons[
					datetime.strptime(item, "%Y-%m-%d").month - 1
					]
			)

		# Replacing week_start_date with seasons columns
		random_DF["seasons"] = pd.Series(seasons_series)
		random_DF = random_DF.drop("week_start_date", axis=1)

		# Converting all columns of the dataframe into a single data type
		random_DF['city'] = random_DF.city.astype("float64")
		random_DF['year'] = random_DF.year.astype("float64")
		random_DF['weekofyear'] = random_DF.weekofyear.astype("float64")
		random_DF['seasons'] = random_DF.seasons.astype("float64")
		if test == False:
			self.dataset = random_DF
		else:
			self.test_dataset = random_DF
		
		# Standardizing the values
		dataframe = pd.DataFrame(self.scaler.fit_transform(random_DF), columns=random_DF.keys())

	def find_columns(self):
		"""
        Calculates the correlation between each of the columns in the dataset and generates a variation of 
		list of columns that lie in a given range.

		:return: None
		"""
		# Compute the correlation of the columns in the dataset
		correlation = self.dataset.corr()
		corr_values = (
			correlation.total_cases
				.drop('total_cases') 
				.sort_values(ascending=False)
		)

		# Generating a list of columns based on varying intervals of the correlation
		corr_max = corr_values.max()
		corr_min = corr_values.min()

		min = 0.10
		max = 0.30

		for i in range(4):
			final_columns = []
			corr_values = corr_values.between(min, max, inclusive=True)

			for item in corr_values.index:
				if corr_values[item] == True:
					final_columns.append(item)

			self.final_columns_list.append(final_columns)
			min -= 0.05
			max += 0.05

	
	def get_column_list(self):
		"""
        Returns the set of columns which produce best predictions
		
		:return: list of columns {}
		"""
		return self.final_columns_list


	def genrate_data_train(self, rows):
		"""
        Given a list of rows, it extracts the data and performs a test train split.
		
		:param rows: List of rows which based on which the models are trained.
		"""
		final_cols = set(self.dataset.keys()) - set(rows)

		# Extracting the test and train data
		modified_y_train = self.dataset['total_cases']
		for col_name in final_cols:
			modified_X_train = self.dataset.drop(col_name, axis=1)

		X_train, X_test, y_train, y_test = train_test_split(
			modified_X_train, modified_y_train, test_size = 0.33, random_state = 42
		)

		# Populating the values for the train
		self.x_vals = modified_X_train.values
		self.y_vals = modified_y_train.values
		self.y_vals = self.y_vals.reshape(self.y_vals.shape[0], 1)

	def generate_data_test(self, rows):
		final_cols = set(self.dataset.keys()) - set(rows)
		for col_name in final_cols:
			modified_X_train = self.dataset.drop(col_name, axis=1)
		self.x_test_vals = modified_X_train.values

	
	# Data Visulaization
	

	def plot_cross_validations(self, x, y):
		"""
        Given the features and the predictions, the cross validation is plotted.
		
		:param x: Features of the data.
		:param x: Predictions of the data.
		:return: None
		"""
		fig, ax = plt.subplots()

		ax.scatter(x, y, edgecolors=(0, 0, 0))
		ax.plot([y.min(), y.max()],
				[y.min(), y.max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		plt.show()

	def beautify_report(self):
		"""
        Formats the final report in a redable format.
		
		:param: None
		:return: None
		"""

		# Header of the report
		head = "Model \t\t\tPrecision \tRecall \t\tF1_Score \tSupport"
		fmt = "{Model:s}\t\t{Precision:0.4f}\t\t{Recall:0.4f}\t\t{F1_Score:0.3f}\t\t{Support:0.3f}"

		print(head)

		# Extracting the relevant data from the final report
		for model in self.report:
			temp = self.report[model]['weighted avg']
			print(fmt.format(
				Model=model,
				Precision=temp['precision'],
				Recall=temp['recall'],
				F1_Score=temp['f1-score'],
				Support=temp['support']
			))

	'''
		Models
	'''

	# Decision tree
	def decision_tree_init(self):
		"""
        Initialization function for the Decision Tree model. 
		
		:param: None
		:return: None
		"""
		self.d_tree.fit(self.x_vals, self.y_vals)

	def decision_tree_predict_train(self):
		"""
        The function trains the Decision Tree model on the train data and updates the reports accordingly.
		
		:param: None
		:return: None
		"""
		self.train_predictions['decision_tree'] = cross_val_predict(self.d_tree, self.x_vals, self.y_vals,
																 cv=10)  # self.d_tree.predict(self.x_vals)
		self.train_predictions['decision_tree'] = self.train_predictions['decision_tree'].astype(int)

		self.report['decision_tree'] = classification_report(
		self.y_vals, self.train_predictions['decision_tree'], output_dict=True
		)

	def decision_tree_predict_test(self):
		"""
        The function predicts the results for the test data based on the train data.

		:param: None
		:return: None
		"""
		self.test_predictions['decision_tree'] = self.d_tree.predict(self.x_test_vals)

	# Support Vectors
	def svc_init(self):
		"""
        Initialization function for the SVM model. 
		
		:param: None
		:return: None
		"""
		self.svc.fit(self.x_vals, self.y_vals)

	def svc_predict_train(self):
		"""
        The function trains the SVM model on the train data and updates the reports accordingly.
		
		:param: None
		:return: None
		"""
		self.train_predictions['svc'] = cross_val_predict(self.svc, self.x_vals, self.y_vals,
																 cv=10)
		#self.svc.predict(self.x_vals)
		self.train_predictions['svc'] = self.train_predictions['svc'].astype(int)

		self.report['svc_train'] = classification_report(
				self.y_vals, self.train_predictions['svc'], output_dict=True
			)


	def svc_predict_test(self):
		"""
        The function predicts the results for the test data based on the train data.

		:param: None
		:return: None
		"""
		self.test_predictions['svc'] = self.svc.predict(self.x_test_vals)

	# Random Forest
	def random_forest_init(self):
		"""
        Initialization function for the Random Forest model. 
		
		:param: None
		:return: None
		"""
		self.rand_forest.fit(self.x_vals, self.y_vals)

	def random_forest_train(self):
		"""
        The function trains the Random Forest model on the train data and updates the reports accordingly.
		
		:param: None
		:return: None
		"""
		self.train_predictions['random_forest'] = cross_val_predict(self.rand_forest, self.x_vals, self.y_vals,
																 cv=10)
		#self.rand_forest.predict(self.x_vals)
		self.train_predictions['random_forest'] = self.train_predictions['random_forest'].astype(int)

		self.report['RF_train'] = classification_report(
			self.y_vals, self.train_predictions['random_forest'], output_dict= True
		)

	def random_forest_test(self):
		"""
        The function predicts the results for the test data based on the train data.

		:param: None
		:return: None
		"""
		self.test_predictions['RF_test'] = self.rand_forest.predict(self.x_test_vals)


	# Adaboost
	def adaboost_init(self):
		"""
        Initialization function for the Adaboost model. 
		
		:param: None
		:return: None
		"""
		self.adaboost.fit(self.x_vals, self.y_vals)

	def adaboost_train(self):
		"""
        The function trains the Adaboost model on the train data and updates the reports accordingly.
		
		:param: None
		:return: None
		"""
		self.train_predictions['adaboost'] = cross_val_predict(self.adaboost, self.x_vals, self.y_vals,
																 cv=10)
		#self.adaboost.predict(self.x_vals)
		self.train_predictions['adaboost'] = self.train_predictions['adaboost'].astype(int)

		self.report['Adaboost_train'] = classification_report(
				self.y_vals, self.train_predictions['adaboost'], output_dict= True
			)

	def adaboost_test(self):
		"""
        The function predicts the results for the test data based on the train data.

		:param: None
		:return: None
		"""
		self.test_predictions['Adaboost_test'] = self.adaboost.predict(self.x_test_vals)


# Driver function
def main():

	# Data Paths
	path_train = ""
	path_train_results = ""
	path_test = ""

	# Report to store the results for the models with best obtained performance
	final_report = {}

	# DengAi Model init.
	deng_object = dengAi(path_train, path_train_results, path_test)
	deng_object.preprocess_data(False)
	deng_object.preprocess_data(True)
	deng_object.find_columns()


	# Train the models and predict the results for a list of columns obtained by 
	# varying intervals of correlation values.
	for row in deng_object.get_column_list():
		if (len(row) != 0):
			deng_object.genrate_data_train(row)
			deng_object.generate_data_test(row)

			# Decision tree
			deng_object.decision_tree_init()
			deng_object.decision_tree_predict_train()
			deng_object.decision_tree_predict_test()
			
			# Support vectors
			deng_object.svc_init()
			deng_object.svc_predict_train()
			deng_object.svc_predict_test()

			# Random Forest
			deng_object.random_forest_init()
			deng_object.random_forest_train()
			deng_object.random_forest_test()

			# Adaboost
			deng_object.adaboost_init()
			deng_object.adaboost_train()
			deng_object.adaboost_test()
	
	# Print the final report
	deng_object.beautify_report()


if __name__ == '__main__':
	main()
