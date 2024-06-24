# PIC 16B Project Summary // Cynthia Du, Meiyi Ye, Nam Truong

> [!IMPORTANT]
> Our files were too large so we could not upload them to GitHub, hence we have decided to share our Google Drive that we used for this project.

#### _Project Overview_
Yelp is a popular platform where users share their experiences and rate businesses on a scale from one to five stars. These reviews significantly influence consumer decisions and play a crucial role in shaping the reputations of businesses. Accurately predicting the ratings of these reviews helps in understanding consumer sentiment and preferences, which is vital for both users and business owners. The ratings must be normalized, as some individuals may rate more generously than others, which would affect the overall rating of the business. Furthermore, when customers are having a bad day, they may tend to rate lower for minor issues. For our project, we developed a predictive model that can determine the _true_ star rating of a Yelp review based on its textual content. By analyzing patterns in word usage and sentiment, the model aims to offer a normalized, bias-free rating that more accurately reflects the actual quality of the service. 

To start our project, we had to conduct extensive data processing to filter review and rating data on restaurants in California. This included merging three datasets, normalizing the data (removing stopwords, etc), and filtering out 'restaurants' as the business type and 'California' as the state. Then, we conducted exploratory data analysis to get a better understanding of our dataset. From there, sentiment analysis was done to understand the most "positive" and "negative" words in the dataset, which was calculated by finding the frequency of the word in positive reviews (3+ stars) minus the frequency of the word in negative reviews (1 or 2 stars). Feature selection using PCA was also conducted to try to minimize the number of words our model had to train through, but results of our models using PCA showed only around 50% accuracy. Thus, we opted to forgo PCA and train our models using our dataset with all of the words as features. 




#### _Access to Necessary Files_

How to set up our environment to run our code:

1. Download our Google Drive Folder [here](https://drive.google.com/drive/folders/1LbtjJctVRZcgwYa3y-uSZ5EigoLwi1Ya?usp=sharing).
2. Upload those files to your personal Google Drive and name the folder `project personal work`.
3. Run our code!



   
#### _Models Used_

Delete later: I think it's better if you try to summarize all the LSTM and CNN together. I listed them all here based on the project report just in case.

1. LSTM Network
   > We developed a Long Short-Term Memory (LSTM) network to predict Yelp review ratings from their text. The model includes an embedding layer that translates words into a 64-dimensional vector space, enhancing semantic understanding. It features two LSTM layers, with 64 and 32 units respectively, to capture and condense the text's contextual dependencies. To prevent overfitting, a 50% dropout rate is used. The network also includes a 64-neuron dense layer with ReLU activation for non-linearity, and a final single-unit linear layer outputs the predicted star rating. The model utilized 687,617 parameters. Unfortunately, it overfitted.
2. LSTM Network with Early Stopping and Dropout Layers
   > We developed an LSTM network to predict Yelp review ratings, incorporating early stopping and dropout layers to handle overfitting and enhance generalization. The model uses an embedding layer with a 10,000-word vocabulary to map words into a 64-dimensional vector space, followed by two LSTM layers with 64 and 32 units for processing and condensing text data. Dropout layers, set at a 50% rate, deactivate neurons randomly to prevent data over-reliance, while a 64-neuron dense layer with ReLU activation adds complexity. Early stopping halted training at the 6th epoch out of 50 planned, upon no further improvement in validation loss, optimizing training efficiency. The entire architecture comprises 687,617 parameters.
3. LSTM network with L2 Regularization
   > To combat overfitting in our LSTM model for Yelp review ratings, we integrated L2 regularization and dropout techniques. L2 regularization, applied to both LSTM and Dense layers with a value of 0.01, penalizes large weights to promote smoother parameter values and generalization. The model includes an Embedding layer, two LSTM layers with regularization, a regularized Dense layer, and a Dropout layer before the final Dense output layer. The model has 687,617 trainable parameters. It did not overfit.
4. Convolutional Neural Network (CNN)
   > We developed a Convolutional Neural Network (CNN) to analyze Yelp review text, utilizing its ability to handle data with spatial hierarchy. The CNN features an Embedding layer for a 10,000-word vocabulary creating 64-dimensional embeddings, followed by a Convolutional layer with 128 filters to capture local text patterns. A Global Max Pooling layer reduces dimensionality by selecting prominent features, and a 64-unit Dense layer with ReLU activation adds non-linearity. To prevent overfitting, a 0.5 Dropout layer is included, ending with a Dense output layer for regression. The model has a total of 689,409 trainable parameters. Unfortunately, it overfitted.
5. CNN with Early Stopping and Dropout Layers
   > To address overfitting in our Yelp review sentiment analysis model, we implemented a CNN with early stopping and dropout techniques. The model includes an embedding layer for a 10,000-word vocabulary, a 1D convolutional layer with 128 filters to capture text sequence patterns, followed by a global max pooling layer for feature summarization. A 64-neuron Dense layer with ReLU activation adds non-linearity, and a 0.5 dropout rate helps prevent overfitting. The model, which has 689,409 parameters, utilizes early stopping to cease training at epoch 8 once no improvement in validation loss is observed, ensuring the model learns generalizable patterns without excessive training.
6. CNN with L2 Regularization
    > In our Yelp review analysis, we utilized a Convolutional Neural Network (CNN) enhanced with L2 regularization to address overfitting. The model features an embedding layer for a 10,000-word vocabulary, a convolutional layer with 128 filters, and a global max pooling layer to reduce dimensionality. It includes two dense layers with ReLU activation and dropout layers with a 0.5 rate to prevent neuron dependency during training. The final output layer uses linear activation for regression tasks, with the model totaling 689,409 trainable parameters. It did not overfit.
7. BERT (Bidirectional Encoder Representations from Transformers)
   > We used DestilBERT which has 66 million parameters. We used 20% of our raw data for the DestilBERT tokenizer with max_length=512. This was about a third of the actual amount of words in each review in the 95th percentile. We had to set the value low to prevent our GPU from running out of memory. It took us about 2 hours to train and as expected, due to this limit, our DestilBERT model did not perform well and overfitted.


#### _Conclusion_
Our project successfully developed a model to predict the star ratings of Yelp reviews based on the text review content. By transforming review text into sequences and applying a trained machine-learning model, we were able to predict star ratings with a reasonable degree of accuracy. Thus, our model provides a valuable tool for both businesses and customers in understanding and analyzing feedback through automated star rating predictions.

# PIC 16B Project Proposal

> [!IMPORTANT]
> You are required to include sections in your proposal that address the following topics. Feel free to include additional sections as needed. Remember that you can create Markdown sections using the # character.

#### _Describe what problem the project addresses, the relation between your project and your background or major, and why you are interested in this project. ( 5 pts )_
> The project aims to predict business ratings based on Yelp data, which provides detailed information about the business and customer reviews. The prediction of business ratings is valuable for both customers and businesses because it allows consumers to make informed decisions on where to dine, and allows businesses to identify areas of improvement for customer satisfaction.
>> **Meiyi Ye** 
>>> I am majoring in Statistics and Data Science, this project is great for applying my knowledge from school to a practical, real world challenge. It involves data manipulation, application of statistical models, machine learning techniques, and possibly natural language processing for handling textual yelp review data. These are all key areas in the field of data science and directly relate to my coursework and training. I am interested in this project because this project allows me to leverage my skills in statistical analysis and machine learning in a context that impacts real businesses and consumers.
>>> 
>>
>> **Cynthia Du**
>>> This project relates to my majors, Statistics + Data Science and Cognitive Science, because I hope to break into the data science industry and provide insights for businesses to implement more sustainable, equitable, and efficient business processes. This project will allow me to gain experience in creating a data science project, and familiarize myself with important data science concepts such as sentiment analysis and machine learning.
>>
>> **Nam Truong**
>>> I am a Mathematics of Computation major. For my career path I have taken an interest in software engineering and data science. I don’t have much exposure to data science so this project will allow me to gain experience. I am interested in this topic because I have always had interest in training some sort of machine learning model to predict some sort of data. More specifically, I live in an area that’s populated with businesses so if this project goes well and my interest takes me beyond this class I can use the end result or skill set from this project to potentially create a catering service for the businesses around me.
    
#### _Describe what resources, such as data, computing power, and etc, you need in order to complete your project. Please pay special attention to the question of data. If your project idea involves data, include at least one link to a data set you can use. It’s also acceptable to link to a website from which you intend to scrape the data you will use (although note that high-quality scraping is a lot of work).  ( 10 pts )_
> We are using a [Yelp open dataset](https://www.yelp.com/dataset) for our project that contains 7 million reviews and 200 thousand pictures for over 150 thousand businesses across 11 metropolitan areas. For our project we plan to do sentiment analysis on users’ reviews, assign it a score, then output the overall sentiment for the business. We will analyze the attributes of the reviews to explain what keywords or phrases that may have a greater contribution to the sentiment score. Since we will be training on a large dataset we will have to do our computations using TPU-4. Additionally, we may use Kaggle/HuggingFace to view similar datasets and draw inspiration on how we can proceed with our project.

#### _Summarize previous work related to your project. For example, if you want to do a data science project using dataset from Kaggle, you should summarize other's work (models, prediction error, and etc. ) ( 5 pts )_
> The [Yelp open dataset](https://www.yelp.com/dataset) has been used for academic purposes as well as dataset challenges that Yelp hosts. For example, a [past winner of the round eight challenge](https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/26e41eb89f65/assets/vendor/pdf/DSC_R08_ClusteredModelAdaptionForPersonalizedSentimentAnalysis.pdf) created a classification model for sentiment analysis. They borrowed ideas found in social psychology and linguistic studies to create a more personalized model that takes into account the individual’s reviews and groups them based on their opinions and how their reviews were expressed. The reviews are grouped based on social comparison theory (how people form groups with others that share similar opinions) and cognitive consistence theory (how members of a group align their opinions with other members). They implemented a model called a Clustered Linear Model Adaptation (cLinAdapt) to estimate the sentiment of the global and group-level adapted models. Overall, their model performed better than user-independent classification –a type of sentiment analysis classification in which the model identifies keywords in the user’s reviews as positive, negative, or neutral and label the review as such based on the overall summed probabilities– as well as several other adaptation models. The key points of this previous can be summarized as: take review, do sentiment analysis, then organize the reviews into matching group sentiment.

#### _Describe required tools and skills. ( 10 pts )_
> We anticipate using Google Cloud to host our project. Scikit-learn, NLTK, Pandas, Matplotlib, Seaborn, TensorFlow/PyTorch, and Jupyter Notebook are some tools that we plan to use. Some skills we may be required to know to achieve our task of making accurate predictions of business ratings are: data preprocessing, data visualization, natural language processing (NLP), machine learning, and various math/statistics concepts.

#### _Describe what you will learn.  ( 5 pts )_
> From our assignments and how our lecture notes are organized, everything is given to us in a step-by-step manner in a Jupyter Notebook, walking us through how to work with some sort of data or code from scratch. This project will teach us how to navigate the Github repository for our dataset and set up our project environment based on the provided instructions on the repository. Our group will use Google Cloud to host our project. Since we do not have any experience using their services we will have to learn how to navigate through it and explore the many tools that are offered to us. 

#### _Include group members and role and tentative timeline for each group member. ( 10 pts )_
> The names of the members that will be contributing to this project are:
>> **Meiyi Ye**
>>> Tentative timeline:
>>> - Week 6
>>>     - Clean the dataset, handle missing values, and format the data for text analysis (tokenization, removing stopwords)
>>> - Week 7
>>>     - EDA (develop features from the text: sentiment scores, frequency of positive and negative words)
>>> - Week 8
>>>     -  Collaborate with team to select and train initial models
>>>     -  Tune models, discuss improvements, model refinement
>>> - Week 9
>>>     -  Prepare final visualizations
>>>     -  Prepare presentation
>>
>> **Cynthia Du**
>>> - Week 5
>>>     - Finalize rough plan of project
>>>     - Familiarize self with data
>>> - Week 6: Data preprocessing
>>>     - Handle NA values
>>>     - Ensure data is in usable format
>>> - Week 7: EDA
>>>     - Generate basic visualizations + tables to explore relationships
>>> - Week 8
>>>     -  Model training, error analysis
>>> - Week 9
>>>     -  Final visualizations, presentation
>>
>> **Nam Truong**
>>> Tentative timeline:
>>> - Week 5
>>>     - Set up Google Cloud and Google Colab
>>>     - Import and analyze data
>>> - Week 6 & 7
>>>     - Preprocess data
>>> - Week 7 & 8
>>>     -  Train model
>>>     -  Model error and analysis
>>> - Week 9
>>>     -  Visualization
>>>     -  Create presentation
