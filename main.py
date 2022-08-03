# Core Pkgs
import re
import string
import streamlit as st 
import pandas as pd 
import pickle
import json
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import requests
import json
import numpy as np

from PIL import Image
from wordcloud import WordCloud

# ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

nltk.download('all')
# Model
svm_model = pickle.load(open("model.sav", 'rb'))
tfidf = pickle.load(open("tfidf.sav", 'rb'))

def getFormText(IDs):
	formText = []
	for formID in IDs:
		apikey = '7f8ce90b8d898a20bebcb12c3d8b52be'
		# Get form questions
		response = requests.get(f"https://api.jotform.com/form/{formID}/questions?apiKey={apikey}")
		responseObject = json.loads(response.content.decode('utf-8'))
		formQuestions = responseObject['content']
		# Get form properties for product text
		response = requests.get(f"https://api.jotform.com/form/{formID}/properties?apiKey={apikey}")    
		responseObject = json.loads(response.content.decode('utf-8'))
		formProperties = responseObject['content']
		text = ""
		for q in formQuestions:
			if "text" in formQuestions[q].keys():
				text += formQuestions[q]["text"].strip() + " "
			if "subHeader" in formQuestions[q].keys():
				text += formQuestions[q]["subHeader"].strip() + " "
			if "subLabel" in formQuestions[q].keys():
				text += formQuestions[q]["subLabel"].strip() + " "
			if "options" in formQuestions[q].keys():
				text += formQuestions[q]["options"].strip() + " "
		
		if "products" in formProperties.keys():
			typeVar = type(formProperties["products"])
			
			if typeVar == dict:
				for p in formProperties["products"]:
					if "name" in formProperties["products"][p].keys():
						text += formProperties["products"][p]["name"].strip() + " "
					if "description" in formProperties["products"][p].keys():
						text += formProperties["products"][p]["description"].strip() + " "
			elif typeVar == list:
				for p in formProperties["products"]:
					if "name" in p.keys():
						text += p["name"].strip() + " "
					if "description" in p.keys():
						text += p["description"].strip() + " "

		formText.append(text)
	
	if len(formText) == 1:
		return formText[0]
	else:
		return formText

#----------------------------------------------- Functions for preprocessing ----------------------------------------
# Removing the html tags
def cleanhtml(text):
	CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	cleantext = re.sub(CLEANR, '', text)
	return cleantext

# Removing the urls
def remove_urls(text):
    cleantext = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text)
    return(cleantext)

#Removing the punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()

# Removing the stop words
def remove_stopwords(text):
	STOPWORDS = set(stopwords.words('english'))
	return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_freqwords(text):
	FREQWORDS = ['name', 'please', 'e', 'mail', 'email','address','number', 'payment','submit', 'phone','date','form','may','us', 'card','example', 'com','yes' , 'no','one','full','like','page','would', 'per','must']
	return " ".join([word for word in str(text).split() if word not in FREQWORDS])

# Remove wors which include digits
def remove_wordswdigit(text):
    cleantext = re.sub("\S*\d\S*", "", text)
    return(cleantext)

# Lemmatize words
def lemmatize_words(text):
	lemmatizer = WordNetLemmatizer()
	return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Check the spelling mistakes
def correct_spellings(text):
	spell = SpellChecker()
	corrected_text = []
	misspelled_words = spell.unknown(text.split())
	for word in text.split():
		if word in misspelled_words:
			corrected_text.append(spell.correction(word))
		else:
			corrected_text.append(word)
	
	return " ".join(corrected_text)
#----------------------------------------------------------------------------------------------------------------------------

def preprocess(formText):
	proc_texts = []
	for text in formText:
		processed = text.lower()
		processed = cleanhtml(processed)
		processed = remove_urls(processed)
		processed = remove_punctuation(processed)
		processed = remove_stopwords(processed)
		processed = remove_freqwords(processed)
		processed = remove_wordswdigit(processed)
		processed = lemmatize_words(processed)
		processed = correct_spellings(processed)
		proc_texts.append(processed)

	if len(proc_texts) == 1:
		return proc_texts[0]
	else:
		return proc_texts
	
def predict(processedText):

	id_to_cluster = {0: 'Application Fee',
						1: 'Course Fee',
						2: 'Deposit Fee',
						3: 'Donation',
						4: 'Membership',
						5: 'Product Fee',
						6: 'Registration Fee',
						7: 'Service Fee',
						8: 'Subscription'}

	if len(processedText) == 1:
		data = {'text': processedText}
		df = pd.DataFrame(data)
		X = tfidf.transform(df["text"]).toarray()
		prediction = svm_model.predict(X)

		probs = svm_model.predict_proba(X)[0]
		data = { 'Clusters' : ['Application Fee','Course Fee','Deposit Fee','Donation','Membership','Product Fee','Registration Fee','Service Fee','Subscription'],
			'Probabilities': probs}
		prob_df = pd.DataFrame(data)

		return id_to_cluster[prediction[0]], prob_df
	else:
		data = {'text': processedText}
		df = pd.DataFrame(data)
		X = tfidf.transform(df["text"]).toarray()
		predictions = svm_model.predict(X)

		result = []
		for p in predictions:
			result.append(id_to_cluster[p])	
		return result


def main():
	menu = ["Predict","Model and Dataset","About"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Predict":
		st.header("Form Payment Classification")
		image = Image.open('cluster.png')
		st.image(image, caption='Clusters')

		menu2 = ["Single Form", "Multiple Forms With File Upload"]
		choice2 = st.selectbox("Select Prediction Type",menu2)

		if choice2 == "Single Form":

			with st.form(key='mlform'):
				with st.container():
					col1, col2 = st.columns(2)
					formID = col1.text_area("Form ID")
					predictButton = col1.form_submit_button(label='Predict')

					if predictButton:
						with st.spinner('This might take a while...'):
							formText = getFormText([formID])
							processed = preprocess([formText])
							result, prob_df = predict([processed])
							col2.text_area("Prediction" , value = result)
				
				with st.container():
					if predictButton:
						col1, col2, col3 = st.columns([1,1.5,1])
						col2.write("--- Probabilities for each Cluster ---")
						col2.dataframe(prob_df.style.highlight_max(color = "darkgoldenrod", subset = ['Probabilities']))

			if predictButton:
				with st.expander("See Form Details"):
					link='[Click to see form](https://www.jotform.com/form/' + formID+')'
					st.markdown(link,unsafe_allow_html=True)

					with st.container():
						col1, col2 = st.columns([1, 3])
						col1.write("Form Text")
						col2.caption(formText)

					st.write("")

					with st.container():
						col1, col2 = st.columns([1,3])
						col1.write("Form Text After Preprocessing")
						col2.caption(processed)

					st.write("")

					with st.container():
						col1, col2 = st.columns([1,3])
						col1.write("Word Cloud")
						wordcloud = WordCloud(
							background_color='white',
							max_words=200,
							max_font_size=40, 
							random_state=42
							).generate(processed)
						st.set_option('deprecation.showPyplotGlobalUse', False)
						plt.imshow(wordcloud, interpolation='bilinear')
						plt.axis("off")
						plt.show()
						col2.pyplot()

		elif choice2 == "Multiple Forms With File Upload":
			
			with st.form(key='mlform2'):
				with st.container():
					col1, col2 = st.columns(2)

					uploaded_file = col1.file_uploader("Choose a txt or csv file")
					predictButton2 = col1.form_submit_button(label='Predict')

					if uploaded_file is not None:
						uploaded = pd.read_csv(uploaded_file, header=None)
				
					if predictButton2 and uploaded_file is not None:
						with st.spinner('This might take a while...'):
							formText = getFormText(np.asarray(uploaded.values.flatten()))
							processed = preprocess(formText)
							result = predict(processed)
							data = { 'Form ID' : np.asarray(uploaded.values.flatten()),
									'Prediction': result
									}
							result_df = pd.DataFrame(data)
							col2.dataframe(result_df)
				
	
	elif choice == "Model and Dataset":
		st.header("Model and Dataset")
		dataAll = pd.read_csv('data2000.csv')

		showData = st.checkbox('Show Dataset')

		if showData:
			st.dataframe(dataAll)

		st.subheader("Number of Samples Per Cluster")	
		st.bar_chart(dataAll['cluster'].value_counts())

		st.subheader("Wordcloud of Training Dataset")
		wordcloud = WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(dataAll['text']))

		
		fig = plt.figure(figsize=(15,8))
		plt.imshow(wordcloud)
		plt.axis('off')
		st.write(fig)

		st.subheader("Most Corrolated Words with Clusters:")
		common = pd.read_csv('commons.csv')
		st.table(common)

		st.subheader("Heat Map of Model")
		hm = Image.open('heatmap.png')
		st.image(hm, caption='Heat Map')

		st.subheader("Classification Report of Model")
		report = pd.read_csv('report.csv')
		st.table(report)
	
	else:
		st.subheader("About")
		st.write("This website is created for the internship project of Jotform. Main aim of the project is classification of the payments conducted in Jotform forms.")
		st.write("Labeling rules are like following")
		st.write("**Application Fee:** Payments that are made only for application itself. Examples: Pet Adoption Application, Rental Application, Program/University Application")
		st.write("**Course Fee:** Payment that are made for course enrollments. Examples: Online Courses, Sport Course, Art Course")
		st.write("**Deposit Fee:** Payments that are made for deposits. Examples: Tattoo Appointment Deposit, Booking Deposit")
		st.write("**Donation:** Payments that are made for donations. Examples: Donation to Church, Donation to Organizations")
		st.write("**Membership:** Payments that are made in order to become a member of any organization. Examples: Club Membership, Association Membership")
		st.write("**Product Fee:** Payments that are made to buy a product. Examples: Buying Tshirt, Buying Food, Buying Gift Card")
		st.write("**Registration Fee:** Payments that are made to get entry to any kind of event. Examples: Camp Registration, Event Tickets")
		st.write("**Service Fee:** Payments that are made in return of a service. Examples: Consultation Service, Medical Service")
		st.write("**Subscription:** Payments that are made weekly-monthly-yearly in return of any service or product. Examples: Magazine Subscription, Parking Place Subscription, Gym Subscription")

if __name__ == '__main__':
	main()