# Utilities import
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Models import
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

# Metrics import
from sklearn.metrics import classification_report

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


class dengAi():
	def __init__(self, train_path, train_results_path, path_test):
		self.X_train = pd.read_csv(
			train_path,
		)
		self.y_train = pd.read_csv(
			train_results_path,
		)

		self.X_test = pd.read_csv(
			path_test,
		)
		self.correlation = []
		self.dataset = []
		self.test_dataset = []
		self.final_columns_list = []

		self.x_vals = []
		self.y_vals = []
		self.x_train_vals = []
		self.y_train_vals = []

		self.init_models()

	def init_models(self):
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
		# Report
		self.report = {}

		self.decision_tree_report = []
		self.svc_report = []
		self.random_forest_report = []
		self.adaboost_report = []
		self.bagging_report = []
		self.knn_report = []

		# Prediction values

		self.decision_tree_predictions_train = []
		self.decision_tree_predictions_test = []

		self.svc_predictions_train = []
		self.svc_predictions_test = []

		self.random_forest_predictions_train = []
		self.random_forest_predictions_test = []

		self.adaboost_predictions_train = []
		self.adaboost_predictions_test = []

		self.bagging_predictions_train = []
		self.bagging_predictions_test = []

		self.knn_predictions_train = []
		self.knn_predictions_test = []


	def preprocess_data(self, test):
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
		correlation = self.dataset.corr()
		corr_values = (
			correlation.total_cases
				.drop('total_cases')  # don't compare with myself
				.sort_values(ascending=False)
		)

		# To get a variation of columns based on the correlation values
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
		return self.final_columns_list


	def genrate_data_train(self, rows):
		final_cols = set(self.dataset.keys()) - set(rows)

		# Extracting the test and train data
		modified_y_train = self.dataset['total_cases']
		for col_name in final_cols:
			modified_X_train = self.dataset.drop(col_name, axis=1)



		X_train, X_test, y_train, y_test = train_test_split(
			modified_X_train, modified_y_train, test_size = 0.33, random_state = 42
		)

		# Populating the values for the train
		self.x_vals = X_train.values
		self.y_vals = y_train.values
		self.y_vals = self.y_vals.reshape(self.y_vals.shape[0], 1)

	def generate_data_test(self, rows):
		final_cols = set(self.dataset.keys()) - set(rows)
		for col_name in final_cols:
			modified_X_train = self.dataset.drop(col_name, axis=1)
		self.x_test_vals = modified_X_train.values

	'''
		Data Visulaization
	'''

	def plot_cross_validations(self, x, y):
		fig, ax = plt.subplots()

		ax.scatter(x, y, edgecolors=(0, 0, 0))
		ax.plot([y.min(), y.max()],
				[y.min(), y.max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		plt.show()



	'''
		Models
	'''

	# Decision tree
	def decision_tree_init(self):
		self.d_tree.fit(self.x_vals, self.y_vals)

	def decision_tree_predict_train(self):
		self.decision_tree_predictions_train = cross_val_predict(self.d_tree, self.x_vals, self.y_vals,
																 cv=10)  # self.d_tree.predict(self.x_vals)
		self.decision_tree_predictions_train = self.decision_tree_predictions_train.astype(int)

		#self.report['DTree_train'] =

		self.decision_tree_report.append(
			classification_report(
				self.y_vals, self.decision_tree_predictions_train, output_dict=True
			)
		)

	def decision_tree_predict_test(self):
		self.report['DTree_test'] = self.d_tree.predict(self.x_test_vals)
		self.decision_tree_predictions_test = self.d_tree.predict(self.x_test_vals)

	# Support Vectors
	def svc_init(self):
		self.svc.fit(self.x_vals, self.y_vals)

	def svc_predict_train(self):
		self.svc_predictions_train = cross_val_predict(self.svc, self.x_vals, self.y_vals,
																 cv=10)
		#self.svc.predict(self.x_vals)
		self.svc_predictions_train = self.svc_predictions_train.astype(int)

		self.report['svc_train'] = classification_report(
				self.y_vals, self.svc_predictions_train, output_dict=True
			)


		self.svc_report.append(
			classification_report(
				self.y_vals, self.svc_predictions_train, output_dict=True
			)
		)

	def svc_predict_test(self):
		self.report['svc_test'] = self.svc.predict(self.x_test_vals)
		self.svc_predictions_test = self.svc.predict(self.x_test_vals)

	# Random Forest
	def random_forest_init(self):
		self.rand_forest.fit(self.x_vals, self.y_vals)

	def random_forest_train(self):
		self.random_forest_predictions_train = cross_val_predict(self.rand_forest, self.x_vals, self.y_vals,
																 cv=10)
		#self.rand_forest.predict(self.x_vals)
		self.random_forest_predictions_train = self.random_forest_predictions_train.astype(int)

		self.report['RF_train'] = classification_report(
				self.y_vals, self.random_forest_predictions_train, output_dict= True
			)


		self.random_forest_report.append(
			classification_report(
				self.y_vals, self.random_forest_predictions_train, output_dict= True
			)
		)

	def random_forest_test(self):
		self.report['RF_test'] = self.rand_forest.predict(self.x_test_vals)
		self.random_forest_predictions_test = self.rand_forest.predict(self.x_test_vals)

	# Adaboost
	def adaboost_init(self):
		self.adaboost.fit(self.x_vals, self.y_vals)

	def adaboost_train(self):
		self.adaboost_predictions_train = cross_val_predict(self.adaboost, self.x_vals, self.y_vals,
																 cv=10)
		#self.adaboost.predict(self.x_vals)
		self.adaboost_predictions_train = self.adaboost_predictions_train.astype(int)

		self.report['Adaboost_train'] = classification_report(
				self.y_vals, self.adaboost_predictions_train, output_dict= True
			)

		self.adaboost_report.append(
			classification_report(
				self.y_vals, self.adaboost_predictions_train, output_dict= True
			)
		)

	def adaboost_test(self):
		self.report['Adaboost_test'] = self.adaboost.predict(self.x_test_vals)
		self.adaboost_predictions_test = self.adaboost.predict(self.x_test_vals)

def main():
	path_train = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_train.csv"
	path_train_results = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_labels_train.csv"
	path_test = "/Users/bhavish96.n/Documents/UTD/Fall '18/Machine Learning [Anurag Nagar]/Assignments/Long Project 1/Project Data/dengue_features_train.csv"

	final_report = {}

	deng_object = dengAi(path_train, path_train_results, path_test)
	deng_object.preprocess_data(False)
	deng_object.preprocess_data(True)
	deng_object.find_columns()



	for row in deng_object.get_column_list():
		if (len(row) != 0):
			deng_object.genrate_data_train(row)
			deng_object.generate_data_test(row)

			# Decision tree
			deng_object.decision_tree_init()
			deng_object.decision_tree_predict_train()
			deng_object.decision_tree_predict_test()
			final_report['Decision_tree'] = deng_object.decision_tree_report[0]['weighted avg']


			# Support vectors
			deng_object.svc_init()
			deng_object.svc_predict_train()
			deng_object.svc_predict_test()
			final_report['SVC'] = deng_object.svc_report[0]['weighted avg']

			# Random Forest
			deng_object.random_forest_init()
			deng_object.random_forest_train()
			deng_object.random_forest_test()
			final_report['Random Forest'] = deng_object.random_forest_report[0]['weighted avg']

			# Adaboost
			deng_object.adaboost_init()
			deng_object.adaboost_train()
			deng_object.adaboost_test()
			final_report['Adaboost'] = deng_object.adaboost_report[0]['weighted avg']

	deng_object.plot_cross_validations(
		deng_object.y_vals,
		deng_object.decision_tree_predictions_train,

	)

	beautify_report(final_report)


def beautify_report(report):
	head = "Model \t\t\t Precision \t\t Recall \t\t F1_Score \t\t Support"
	fmt = "{Model:s}\t\t{Precision:0.4f}\t\t{Recall:0.4f}\t\t{F1_Score:0.3f}\t\t{Support:0.3f}"

	print(head)
	for record in report:
		temp = report[record]
		print(fmt.format(
			Model=record,
			Precision=temp['precision'],
			Recall=temp['recall'],
			F1_Score=temp['f1-score'],
			Support=temp['support']
		))

	'''
	print ("\n")
	print ("\t Model \t\t Precision \t\t Recall \t\t F1-Score \t\t Support")
	for record in report:
		temp = report[record]
		print ("\t", record, "\t %.3f \t %.3f \t %.3f \t %.3f" % (temp['precision'], temp['recall'], temp['f1-score'], temp['support']))
	'''


if __name__ == '__main__':
	main()
